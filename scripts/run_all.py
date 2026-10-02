#!/usr/bin/env python3
"""Run the full SGO-Audit pipeline end-to-end and publish a tracked summary.

Usage:
    python3 scripts/run_all.py
    python3 scripts/run_all.py --current
    python3 scripts/run_all.py --llm-sample 50 --ollama-model llama3.2:1b
    python3 scripts/run_all.py --dry-run-llm   # skip real Ollama

Writes a human-readable summary to results/RUN_SUMMARY.md plus copied key
JSON/CSV excerpts under results/ so findings are visible on GitHub without
local tooling.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=False,
        check=False,
    )
    if check and completed.returncode != 0:
        raise SystemExit(f"Command failed ({completed.returncode}): {' '.join(cmd)}")
    return completed


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_num(value: object, digits: int = 3) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.{digits}f}"
    return "n/a" if value is None else str(value)


def parse_vintage_label(raw: str | None) -> str:
    """Normalize a DATA_VINTAGE.txt first line or free-form label."""

    if not raw:
        return "unknown"
    text = raw.strip()
    if text.startswith("DATA_VINTAGE="):
        text = text.split("=", 1)[1].strip()
    return text or "unknown"


def count_reference_report_ids(reference: dict) -> int:
    """Count unique Phase 3 report IDs (strings or ``{"report_id": ...}`` objects).

    The curated ``nhtsa_investigation_report_ids.json`` mixes plain string IDs
    (PE24031) with object entries that carry per-ID notes (EA26002). Naively
    doing ``set.update(report_ids)`` raises ``TypeError: unhashable type: 'dict'``
    and previously crashed ``build_summary`` before ``results/`` could publish.
    """

    all_ids: set[str] = set()
    for inv in reference.get("investigations", []):
        if not isinstance(inv, dict):
            continue
        for item in inv.get("report_ids", []):
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    all_ids.add(cleaned)
            elif isinstance(item, dict) and item.get("report_id"):
                all_ids.add(str(item["report_id"]).strip())
    if all_ids:
        return len(all_ids)
    # Fallbacks for alternate shapes in a validation_report snapshot.
    if "report_ids" in reference:
        return len(reference["report_ids"])
    return len(reference.get("investigations", []))


def resolve_data_vintage(
    *,
    ingest_metadata: dict | None = None,
    vintage_file: Path | None = None,
    fallback: str | None = None,
) -> str:
    """Prefer the run-level ingest ``data_vintage`` field (TECHNICAL_IMPLEMENTATION 3.1)."""

    if ingest_metadata:
        labeled = ingest_metadata.get("data_vintage")
        if labeled:
            return parse_vintage_label(str(labeled))
    if vintage_file is not None and vintage_file.exists():
        first = vintage_file.read_text(encoding="utf-8").splitlines()
        if first:
            return parse_vintage_label(first[0])
    return parse_vintage_label(fallback)


def copy_if_exists(src: Path, dest: Path) -> None:
    if src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def current_git_branch(default: str = "main") -> str:
    """Return the current git branch name, or ``default`` if detection fails."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return default
    branch = completed.stdout.strip()
    if completed.returncode != 0 or not branch or branch == "HEAD":
        return default
    return branch


def ollama_available(url: str = "http://localhost:11434/api/tags") -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def build_summary(
    *,
    vintage: str,
    llm_backend: str,
    llm_sample: int,
    ollama_model: str,
) -> str:
    ingest = load_json(ROOT / "data/processed/ingest_metadata.json")
    consensus = load_json(ROOT / "data/processed/consensus_summary.json")
    operating = load_json(ROOT / "data/processed/classifier_operating_point.json")
    metrics = load_json(ROOT / "data/processed/classifier_metrics.json")
    disagreement = load_json(ROOT / "data/processed/disagreement_matrix.json")
    validation = load_json(ROOT / "data/processed/validation_report.json")
    # Surface the ingest run-level vintage (not just the CLI/download label).
    vintage = resolve_data_vintage(
        ingest_metadata=ingest,
        vintage_file=ROOT / "data/raw/DATA_VINTAGE.txt",
        fallback=vintage,
    )
    snapshot = validation.get("current_data_snapshot", {})
    flag_sets = validation.get("flag_sets", [])
    overlaps = {
        item.get("name", "unknown"): item.get("reference_id_overlap_count", 0)
        for item in flag_sets
        if isinstance(item, dict)
    }
    present_count = snapshot.get(
        "reference_ids_present_count",
        validation.get("reference_ids_present_count", "n/a"),
    )
    missing_ids = snapshot.get(
        "reference_ids_missing",
        validation.get("reference_ids_missing", []),
    )
    reference_count = len(validation.get("reference", {}).get("investigations", []))
    if "report_ids" in validation.get("reference", {}):
        reference_count = len(validation["reference"]["report_ids"])
    # Prefer explicit total from the curated reference JSON.
    # report_ids entries may be plain strings or {"report_id": "..."} objects.
    ref_path = ROOT / "data/reference/nhtsa_investigation_report_ids.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        counted = count_reference_report_ids(ref)
        if counted:
            reference_count = counted

    heuristic_flags = ROOT / "data/processed/heuristic_flags.csv"
    classifier_flags = ROOT / "data/processed/classifier_flagged_records.csv"
    heuristic_n = (
        sum(1 for _ in heuristic_flags.open(encoding="utf-8")) - 1
        if heuristic_flags.exists()
        else 0
    )
    classifier_n = (
        sum(1 for _ in classifier_flags.open(encoding="utf-8")) - 1
        if classifier_flags.exists()
        else 0
    )

    ads_rows = ingest.get("inputs", {}).get("ads", {}).get("rows_clean", "?")
    adas_rows = ingest.get("inputs", {}).get("adas", {}).get("rows_clean", "?")
    combined = ads_rows if isinstance(ads_rows, int) and isinstance(adas_rows, int) else None
    if combined is not None:
        combined = ads_rows + adas_rows

    op = operating.get("operating_point", {})
    counts = disagreement.get("counts", disagreement.get("category_counts", {}))
    if not counts and isinstance(disagreement.get("matrix"), dict):
        counts = disagreement["matrix"]
    calibration = metrics.get("calibration", "none")

    lines = [
        "# SGO-Audit end-to-end run summary",
        "",
        f"Generated (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"Data vintage: **{vintage}**",
        f"LLM backend: **{llm_backend}**",
        f"LLM sample size: **{llm_sample}**",
        f"Ollama model requested: `{ollama_model}`",
        "",
        "## Branch / merge state",
        "",
        f"- Integration base going forward: `{current_git_branch()}`.",
        "- Merged: PR #2 (`AGENTS.md` setup docs) + PR #3 (consensus queue +",
        "  optional `--calibration`).",
        "- Out of scope for this consolidate pass (next, not now): narrative-text",
        "  Phase 1 features, Phase 2 strengthening, Phase 3 ground-truth expansion.",
        "",
        "## Phase 0 — ingest + heuristics",
        f"- ADS cleaned rows: `{ads_rows}`",
        f"- ADAS cleaned rows: `{adas_rows}`",
        f"- Combined (approx): `{combined if combined is not None else 'see ingest_metadata.json'}`",
        f"- Heuristic-flagged records: `{heuristic_n}`",
        "",
        "## Phase 1 — classical classifier",
        f"- Selected flag model: `{operating.get('selected_flag_model', metrics.get('selected_flag_model', 'n/a'))}`",
        f"- Probability calibration: `{calibration}` (default `none`; see decision below)",
        f"- Operating threshold: `{_fmt_num(op.get('threshold'))}`",
        f"- Synthetic-flip precision / recall: `{_fmt_num(op.get('precision'))}` / `{_fmt_num(op.get('recall'))}`",
        f"- Model disagreement flags: `{classifier_n}`",
        f"- Rationale: {op.get('rationale', 'n/a')}",
        "",
        "### Precision-first threshold + calibration default",
        "",
        "- Threshold selection is precision-first in `src/classifier.py`",
        "  (`choose_operating_point`): require synthetic precision ≥ 80%, then",
        "  maximize recall within that constraint.",
        "- `--calibration` default reviewed against Archive-2021-2025:",
        "  `none` ≈ `isotonic` precision (~0.91) with fewer, higher-confidence",
        "  flags; `sigmoid`/`isotonic` expanded the queue at lower mean",
        "  confidence. **Default remains `none`** (opt-in exploration only).",
        "",
        "## Phase 2 — narrative auditor + disagreement matrix",
        f"- Backend used: `{llm_backend}`",
        f"- Disagreement counts: `{json.dumps(counts)}`",
        "",
        "## Consensus review queue (Phase 0/1/2 fusion)",
        f"- Records in queue: `{consensus.get('records_in_queue', 'n/a')}`",
        f"- Tier counts: `{json.dumps(consensus.get('tier_counts', {}))}`",
        f"- Records with Phase 2 evidence: `{consensus.get('records_with_phase2_evidence', 'n/a')}`",
        "",
        "## Phase 3 — NHTSA investigation ID cross-reference",
        f"- Reference IDs: `{reference_count}`",
        f"- Present in snapshot: `{present_count}`",
        f"- Missing from snapshot: `{', '.join(missing_ids) if missing_ids else 'none'}`",
        f"- Flag overlaps: `{json.dumps(overlaps)}`",
        "",
        "## Responsible use",
        "",
        "Flags are review signals only — not confirmed misclassifications, not",
        "manufacturer safety rankings, and not exposure-normalized crash rates.",
        "",
        "## Artifacts in this folder",
        "",
        "- `ingest_metadata.json`",
        "- `classifier_operating_point.json`",
        "- `classifier_metrics.json`",
        "- `disagreement_matrix.json`",
        "- `validation_report.json`",
        "- `heuristic_flags.csv`",
        "- `classifier_flagged_records.csv`",
        "- `consensus_review_queue.csv`",
        "- `consensus_summary.json`",
        "- `validation_note.md`",
        "",
    ]
    return "\n".join(lines)


def publish_results(*, vintage: str, llm_backend: str, llm_sample: int, ollama_model: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    copy_if_exists(ROOT / "data/processed/ingest_metadata.json", RESULTS / "ingest_metadata.json")
    copy_if_exists(
        ROOT / "data/processed/classifier_operating_point.json",
        RESULTS / "classifier_operating_point.json",
    )
    copy_if_exists(ROOT / "data/processed/classifier_metrics.json", RESULTS / "classifier_metrics.json")
    copy_if_exists(ROOT / "data/processed/disagreement_matrix.json", RESULTS / "disagreement_matrix.json")
    copy_if_exists(ROOT / "data/processed/validation_report.json", RESULTS / "validation_report.json")
    copy_if_exists(ROOT / "data/processed/heuristic_flags.csv", RESULTS / "heuristic_flags.csv")
    copy_if_exists(
        ROOT / "data/processed/classifier_flagged_records.csv",
        RESULTS / "classifier_flagged_records.csv",
    )
    copy_if_exists(ROOT / "outputs/validation_note.md", RESULTS / "validation_note.md")
    copy_if_exists(ROOT / "data/processed/phase_comparison.csv", RESULTS / "phase_comparison.csv")
    copy_if_exists(ROOT / "data/processed/consensus_review_queue.csv", RESULTS / "consensus_review_queue.csv")
    copy_if_exists(ROOT / "data/processed/consensus_summary.json", RESULTS / "consensus_summary.json")
    summary = build_summary(
        vintage=vintage,
        llm_backend=llm_backend,
        llm_sample=llm_sample,
        ollama_model=ollama_model,
    )
    (RESULTS / "RUN_SUMMARY.md").write_text(summary, encoding="utf-8")
    print(f"\nPublished tracked results under {RESULTS}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SGO-Audit end-to-end and publish results/")
    parser.add_argument("--current", action="store_true", help="Use current third-amended CSVs")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--llm-sample", type=int, default=50)
    parser.add_argument("--ollama-model", default="llama3.2:1b")
    parser.add_argument(
        "--llm-backend",
        choices=["auto", "ollama", "transformers", "dry-run"],
        default="transformers",
        help=(
            "Phase 2 backend. Default 'transformers' is reliable on CPU VMs where "
            "Ollama's llama-server may segfault. Use 'ollama' on a local GPU host."
        ),
    )
    parser.add_argument(
        "--transformers-model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Hugging Face model id when --llm-backend=transformers",
    )
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)

    py = sys.executable

    if not args.skip_download:
        download_cmd = [py, "scripts/download_data.py", "--overwrite"]
        if args.current:
            download_cmd.append("--current")
        run(download_cmd)

    vintage_path = ROOT / "data/raw/DATA_VINTAGE.txt"
    vintage = resolve_data_vintage(vintage_file=vintage_path)

    run([py, "-m", "src.ingest"])
    run([py, "-m", "src.heuristics"])
    run([py, "-m", "src.entity_trends"])
    run([py, "-m", "src.classifier"])

    if args.llm_backend == "dry-run":
        llm_backend = "heuristic_fallback"
        llm_cmd = [
            py,
            "-m",
            "src.llm_auditor",
            "--sample",
            str(args.llm_sample),
            "--dry-run",
        ]
        ollama_model = args.ollama_model
    elif args.llm_backend == "transformers":
        llm_backend = "transformers"
        ollama_model = args.transformers_model
        llm_cmd = [
            py,
            "-m",
            "src.llm_auditor",
            "--sample",
            str(args.llm_sample),
            "--backend",
            "transformers",
            "--model",
            args.transformers_model,
            "--no-fallback",
        ]
    else:
        use_real_llm = ollama_available()
        ollama_model = args.ollama_model
        llm_cmd = [
            py,
            "-m",
            "src.llm_auditor",
            "--sample",
            str(args.llm_sample),
            "--backend",
            args.llm_backend,
            "--model",
            args.ollama_model,
        ]
        if use_real_llm and args.llm_backend in {"auto", "ollama"}:
            llm_backend = "ollama"
        else:
            llm_cmd.append("--dry-run")
            llm_backend = "heuristic_fallback"
            print(
                "\nOllama unavailable; Phase 2 will use labeled heuristic_fallback.",
                flush=True,
            )
    run(llm_cmd)
    run([py, "-m", "src.compare"])
    run([py, "-m", "src.consensus"])
    run([py, "-m", "src.validate"])

    if not args.skip_tests:
        run([py, "-m", "pytest", "-q"])

    publish_results(
        vintage=vintage,
        llm_backend=llm_backend,
        llm_sample=args.llm_sample,
        ollama_model=ollama_model,
    )
    print("\nDONE. Open results/RUN_SUMMARY.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
