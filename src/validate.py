"""Phase 3 NHTSA investigation cross-reference validation.

This module checks whether project flag sets overlap with a small, manually
curated set of SGO Report IDs named in published ODI investigation resumes.
The IDs are not ground-truth misclassification labels; overlap is only a
directional signal that project consistency flags touch records NHTSA has
already scrutinized.

Usage:
    python -m src.validate

TODO: expand the curated Phase 3 reference set beyond the current
PE24031 / EA26002 IDs as new NHTSA investigation materials are published.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .fields import REPORT_ID


DEFAULT_REFERENCE_PATH = Path("data/reference/nhtsa_investigation_report_ids.json")
DEFAULT_COMBINED_FRAME_PATH = Path("data/processed/combined_audit_frame.csv")
DEFAULT_HEURISTIC_FLAGS_PATH = Path("data/processed/heuristic_flags.csv")
DEFAULT_CLASSIFIER_FLAGS_PATH = Path("data/processed/classifier_flagged_records.csv")
DEFAULT_PHASE_COMPARISON_PATH = Path("data/processed/phase_comparison.csv")
DEFAULT_INGEST_METADATA_PATH = Path("data/processed/ingest_metadata.json")
DEFAULT_REPORT_PATH = Path("data/processed/validation_report.json")
DEFAULT_NOTE_PATH = Path("outputs/validation_note.md")

HIGH_PRIORITY_COMPARISON_CATEGORY = "phase1_phase2_agree_vs_reported"

RESPONSIBLE_USE_NOTE = (
    "Flags are review signals only, not findings of misclassification, fault, "
    "or comparative safety performance."
)
VALIDATION_FRAMING = (
    "Directional validation only: these ODI investigation IDs identify crashes "
    "NHTSA scrutinized for FSD reduced-visibility performance, not labeled "
    "ground truth for automation-level misclassification."
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype="string", keep_default_na=False, na_values=[])


def _normalize_report_id(value: object) -> str | None:
    """Return a stripped Report ID string, or None for blank/missing values."""

    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _sorted_ids(ids: set[str]) -> list[str]:
    return sorted(ids)


def extract_reference_ids(reference: dict[str, Any]) -> tuple[set[str], dict[str, list[dict[str, str]]]]:
    """Extract unique Report IDs and their source references from curated JSON.

    ``report_ids`` entries may be plain strings or small objects containing a
    ``report_id`` key plus optional per-ID notes.
    """

    all_ids: set[str] = set()
    sources_by_id: dict[str, list[dict[str, str]]] = {}
    for investigation in reference.get("investigations", []):
        source = {
            "investigation_name": str(investigation.get("investigation_name", "")),
            "url": str(investigation.get("url", "")),
            "notes": str(investigation.get("notes", "")),
        }
        for entry in investigation.get("report_ids", []):
            if isinstance(entry, dict):
                report_id = _normalize_report_id(entry.get("report_id"))
                notes = str(entry.get("notes", ""))
            else:
                report_id = _normalize_report_id(entry)
                notes = ""
            if report_id is None:
                continue
            all_ids.add(report_id)
            source_entry = source.copy()
            if notes:
                source_entry["report_id_notes"] = notes
            sources_by_id.setdefault(report_id, []).append(source_entry)
    return all_ids, sources_by_id


def load_reference(path: Path) -> tuple[dict[str, Any], set[str], dict[str, list[dict[str, str]]]]:
    """Load the manually curated investigation reference artifact."""

    reference = json.loads(path.read_text(encoding="utf-8"))
    reference_ids, sources_by_id = extract_reference_ids(reference)
    if not reference_ids:
        raise ValueError(f"Reference file {path} did not contain any report IDs.")
    return reference, reference_ids, sources_by_id


def report_id_set(frame: pd.DataFrame, *, source: str) -> set[str]:
    """Return unique nonblank Report IDs from ``frame``."""

    if REPORT_ID not in frame.columns:
        raise ValueError(f"{source} must include {REPORT_ID!r}.")
    return {
        report_id
        for report_id in frame[REPORT_ID].map(_normalize_report_id)
        if report_id is not None
    }


def load_report_ids_from_csv(path: Path, *, source_name: str) -> tuple[set[str], int]:
    """Load unique Report IDs and row count from a CSV artifact."""

    frame = _read_csv(path)
    return report_id_set(frame, source=source_name), int(len(frame))


def load_phase_comparison_flag_ids(
    path: Path,
    *,
    category: str = HIGH_PRIORITY_COMPARISON_CATEGORY,
) -> tuple[set[str], int]:
    """Load Report IDs for the requested Phase 2 comparison category."""

    frame = _read_csv(path)
    if "disagreement_category" not in frame.columns:
        raise ValueError(f"{path} must include 'disagreement_category'.")
    selected = frame.loc[frame["disagreement_category"].astype("string").str.strip() == category]
    return report_id_set(selected, source=str(path)), int(len(selected))


def summarize_flag_set(
    *,
    name: str,
    path: Path,
    ids: set[str],
    records_loaded: int,
    reference_ids: set[str],
    snapshot_present_reference_ids: set[str],
    status: str = "loaded",
    notes: str = "",
) -> dict[str, Any]:
    """Build a JSON-safe overlap summary for one flag set."""

    reference_overlap = ids & reference_ids
    snapshot_overlap = ids & snapshot_present_reference_ids
    return {
        "name": name,
        "path": str(path),
        "status": status,
        "records_loaded": records_loaded,
        "distinct_report_ids": len(ids),
        "reference_id_overlap_count": len(reference_overlap),
        "reference_id_overlap_ids": _sorted_ids(reference_overlap),
        "snapshot_present_reference_id_overlap_count": len(snapshot_overlap),
        "snapshot_present_reference_id_overlap_ids": _sorted_ids(snapshot_overlap),
        "notes": notes,
    }


def summarize_optional_csv_flag_set(
    *,
    name: str,
    path: Path,
    reference_ids: set[str],
    snapshot_present_reference_ids: set[str],
    notes: str,
) -> dict[str, Any]:
    """Summarize a flag CSV when present; otherwise return a missing-file row."""

    if not path.exists():
        return summarize_flag_set(
            name=name,
            path=path,
            ids=set(),
            records_loaded=0,
            reference_ids=reference_ids,
            snapshot_present_reference_ids=snapshot_present_reference_ids,
            status="missing",
            notes=f"{notes} File was not present, so this overlap was not evaluated.",
        )
    ids, records_loaded = load_report_ids_from_csv(path, source_name=name)
    return summarize_flag_set(
        name=name,
        path=path,
        ids=ids,
        records_loaded=records_loaded,
        reference_ids=reference_ids,
        snapshot_present_reference_ids=snapshot_present_reference_ids,
        notes=notes,
    )


def summarize_optional_phase_comparison(
    *,
    path: Path,
    reference_ids: set[str],
    snapshot_present_reference_ids: set[str],
    category: str = HIGH_PRIORITY_COMPARISON_CATEGORY,
) -> dict[str, Any]:
    """Summarize the optional high-priority Phase 1/Phase 2 agreement category."""

    notes = f"Optional Phase 2 comparison subset where disagreement_category == {category!r}."
    if not path.exists():
        return summarize_flag_set(
            name="phase1_phase2_agree_vs_reported",
            path=path,
            ids=set(),
            records_loaded=0,
            reference_ids=reference_ids,
            snapshot_present_reference_ids=snapshot_present_reference_ids,
            status="missing",
            notes=f"{notes} File was not present, so this overlap was not evaluated.",
        )
    ids, records_loaded = load_phase_comparison_flag_ids(path, category=category)
    return summarize_flag_set(
        name="phase1_phase2_agree_vs_reported",
        path=path,
        ids=ids,
        records_loaded=records_loaded,
        reference_ids=reference_ids,
        snapshot_present_reference_ids=snapshot_present_reference_ids,
        notes=notes,
    )


def load_ingest_metadata(path: Path) -> dict[str, Any] | None:
    """Load optional data-vintage metadata when available."""

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_validation_report(
    *,
    reference_path: Path = DEFAULT_REFERENCE_PATH,
    combined_frame_path: Path = DEFAULT_COMBINED_FRAME_PATH,
    heuristic_flags_path: Path = DEFAULT_HEURISTIC_FLAGS_PATH,
    classifier_flags_path: Path = DEFAULT_CLASSIFIER_FLAGS_PATH,
    phase_comparison_path: Path = DEFAULT_PHASE_COMPARISON_PATH,
    ingest_metadata_path: Path = DEFAULT_INGEST_METADATA_PATH,
) -> dict[str, Any]:
    """Build the Phase 3 validation report without writing files."""

    reference, reference_ids, sources_by_id = load_reference(reference_path)
    combined = _read_csv(combined_frame_path)
    snapshot_ids = report_id_set(combined, source=str(combined_frame_path))
    snapshot_present_reference_ids = reference_ids & snapshot_ids
    snapshot_missing_reference_ids = reference_ids - snapshot_ids

    flag_sets = [
        summarize_optional_csv_flag_set(
            name="heuristic_flags",
            path=heuristic_flags_path,
            reference_ids=reference_ids,
            snapshot_present_reference_ids=snapshot_present_reference_ids,
            notes="Phase 0 rule-based consistency flags.",
        ),
        summarize_optional_csv_flag_set(
            name="classifier_flagged_records",
            path=classifier_flags_path,
            reference_ids=reference_ids,
            snapshot_present_reference_ids=snapshot_present_reference_ids,
            notes="Phase 1 high-confidence classifier/reported-label disagreements.",
        ),
        summarize_optional_phase_comparison(
            path=phase_comparison_path,
            reference_ids=reference_ids,
            snapshot_present_reference_ids=snapshot_present_reference_ids,
        ),
    ]

    expected_present = set(reference.get("archive_2021_2025_expected_present_ids", []))
    expected_missing = set(reference.get("archive_2021_2025_expected_missing_ids", []))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_framing": VALIDATION_FRAMING,
        "responsible_use": RESPONSIBLE_USE_NOTE,
        "reference": {
            "path": str(reference_path),
            "description": reference.get("description", ""),
            "reference_id_count": len(reference_ids),
            "reference_ids": _sorted_ids(reference_ids),
            "sources_by_report_id": {report_id: sources_by_id[report_id] for report_id in _sorted_ids(reference_ids)},
            "investigations": [
                {
                    "investigation_name": investigation.get("investigation_name", ""),
                    "url": investigation.get("url", ""),
                    "notes": investigation.get("notes", ""),
                    "report_id_count": len(investigation.get("report_ids", [])),
                }
                for investigation in reference.get("investigations", [])
            ],
            "archive_2021_2025_expected_present_ids": _sorted_ids(expected_present),
            "archive_2021_2025_expected_missing_ids": _sorted_ids(expected_missing),
        },
        "current_data_snapshot": {
            "path": str(combined_frame_path),
            "records_loaded": int(len(combined)),
            "distinct_report_ids": len(snapshot_ids),
            "reference_ids_present_count": len(snapshot_present_reference_ids),
            "reference_ids_present": _sorted_ids(snapshot_present_reference_ids),
            "reference_ids_missing_count": len(snapshot_missing_reference_ids),
            "reference_ids_missing": _sorted_ids(snapshot_missing_reference_ids),
            "expected_present_ids_not_found": _sorted_ids(expected_present - snapshot_ids),
            "expected_missing_ids_found": _sorted_ids(expected_missing & snapshot_ids),
            "ingest_metadata": load_ingest_metadata(ingest_metadata_path),
        },
        "flag_sets": flag_sets,
        "caveats": [
            "The validation sample is small and manually extracted from published ODI resumes.",
            "The named ODI records are crashes scrutinized for FSD reduced-visibility performance, not labels proving automation-level misclassification.",
            "Overlap between flags and ODI-named records is a directional review signal, not statistical proof.",
            "Missing recent IDs may reflect the current Archive-2021-2025 data snapshot vintage rather than an error.",
        ],
    }


def render_validation_note(report: dict[str, Any]) -> str:
    """Render a short human-readable validation note."""

    snapshot = report["current_data_snapshot"]
    reference = report["reference"]
    lines = [
        "# Phase 3 validation note",
        "",
        report["validation_framing"],
        "",
        "## Snapshot presence",
        "",
        (
            f"- {snapshot['reference_ids_present_count']} of "
            f"{reference['reference_id_count']} reference SGO Report IDs appear in the "
            "current combined audit frame."
        ),
        (
            f"- Missing from this snapshot: "
            f"{', '.join(snapshot['reference_ids_missing']) if snapshot['reference_ids_missing'] else 'none'}."
        ),
        "",
        "## Flag-set overlap",
        "",
    ]
    for flag_set in report["flag_sets"]:
        lines.append(
            f"- {flag_set['name']}: "
            f"{flag_set['snapshot_present_reference_id_overlap_count']} of "
            f"{snapshot['reference_ids_present_count']} snapshot-present reference IDs; "
            f"{flag_set['reference_id_overlap_count']} of {reference['reference_id_count']} total reference IDs. "
            f"Status: {flag_set['status']}."
        )
    lines.extend(
        [
            "",
            "## Caveats and responsible use",
            "",
            "- This is a small validation sample; do not treat it as statistical proof.",
            "- Flags are not findings. They identify records for human review only.",
            "- The ODI resumes scrutinized these crashes for FSD reduced-visibility performance; they do not label the records as automation-level misclassifications.",
            "- No output here supports manufacturer safety-rate or comparative safety claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_validation(
    *,
    reference_path: Path = DEFAULT_REFERENCE_PATH,
    combined_frame_path: Path = DEFAULT_COMBINED_FRAME_PATH,
    heuristic_flags_path: Path = DEFAULT_HEURISTIC_FLAGS_PATH,
    classifier_flags_path: Path = DEFAULT_CLASSIFIER_FLAGS_PATH,
    phase_comparison_path: Path = DEFAULT_PHASE_COMPARISON_PATH,
    ingest_metadata_path: Path = DEFAULT_INGEST_METADATA_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    note_path: Path = DEFAULT_NOTE_PATH,
) -> dict[str, Any]:
    """Run validation and write JSON plus Markdown artifacts."""

    report = build_validation_report(
        reference_path=reference_path,
        combined_frame_path=combined_frame_path,
        heuristic_flags_path=heuristic_flags_path,
        classifier_flags_path=classifier_flags_path,
        phase_comparison_path=phase_comparison_path,
        ingest_metadata_path=ingest_metadata_path,
    )
    report["validation_report_path"] = str(report_path)
    report["validation_note_path"] = str(note_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    note_path.write_text(render_validation_note(report), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for ``python -m src.validate``."""

    parser = argparse.ArgumentParser(description="Cross-reference SGO flags against ODI-named Report IDs.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--combined-frame", type=Path, default=DEFAULT_COMBINED_FRAME_PATH)
    parser.add_argument("--heuristic-flags", type=Path, default=DEFAULT_HEURISTIC_FLAGS_PATH)
    parser.add_argument("--classifier-flags", type=Path, default=DEFAULT_CLASSIFIER_FLAGS_PATH)
    parser.add_argument("--phase-comparison", type=Path, default=DEFAULT_PHASE_COMPARISON_PATH)
    parser.add_argument("--ingest-metadata", type=Path, default=DEFAULT_INGEST_METADATA_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--note-output", type=Path, default=DEFAULT_NOTE_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = run_validation(
        reference_path=args.reference,
        combined_frame_path=args.combined_frame,
        heuristic_flags_path=args.heuristic_flags,
        classifier_flags_path=args.classifier_flags,
        phase_comparison_path=args.phase_comparison,
        ingest_metadata_path=args.ingest_metadata,
        report_path=args.report_output,
        note_path=args.note_output,
    )
    summary = {
        "reference_id_count": report["reference"]["reference_id_count"],
        "reference_ids_present_count": report["current_data_snapshot"]["reference_ids_present_count"],
        "reference_ids_missing": report["current_data_snapshot"]["reference_ids_missing"],
        "flag_set_overlaps": {
            flag_set["name"]: flag_set["snapshot_present_reference_id_overlap_count"]
            for flag_set in report["flag_sets"]
        },
        "validation_report_path": report["validation_report_path"],
        "validation_note_path": report["validation_note_path"],
        "responsible_use": report["responsible_use"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
