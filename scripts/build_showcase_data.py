#!/usr/bin/env python3
"""Build showcase/data.js from tracked results/ artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "showcase" / "data.js"

RULE_PLAIN = {
    "ads_file_but_adas_engaged": (
        "Filed as ADS, but the engagement field looks like Level 2 ADAS."
    ),
    "adas_file_but_ads_engaged": (
        "Filed as Level 2 ADAS, but the engagement field looks like ADS."
    ),
    "ads_label_but_consumer_driver": (
        "Labeled ADS, but driver/operator type looks like a consumer driver."
    ),
    "adas_label_but_no_driver": (
        "Labeled Level 2 ADAS, but no driver/operator is indicated."
    ),
    "ads_narrative_level2_language": (
        "ADS label, but the narrative uses Level-2 driver-supervision language."
    ),
}


def _load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _load_csv(name: str) -> list[dict]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _trim_queue(row: dict) -> dict:
    conf = row.get("phase1_prediction_confidence") or ""
    return {
        "report_id": row["Report ID"],
        "entity": row["Reporting Entity"].strip('"'),
        "reported": row["reported_automation_level"],
        "score": int(float(row["consensus_score"])),
        "tier": row["consensus_tier"],
        "heuristic": row["heuristic_disagrees"] == "True",
        "phase1": row["phase1_disagrees"] == "True",
        "phase2": row["phase2_disagrees"] == "True",
        "phase1_pred": row.get("phase1_predicted_automation_level") or None,
        "phase1_conf": float(conf) if conf else None,
        "phase2_pred": row.get("phase2_llm_inferred_automation_level") or None,
        "rules": row.get("heuristic_rules_triggered") or "",
        "category": row.get("phase2_disagreement_category") or "",
    }


def _trim_clf(row: dict) -> dict:
    return {
        "report_id": row["Report ID"],
        "entity": row["Reporting Entity"].strip('"'),
        "reported": row["reported_automation_level"],
        "predicted": row["predicted_automation_level"],
        "confidence": float(row["disagreement_confidence"]),
    }


def _plain_rules(rules: str) -> str:
    parts = [p.strip() for p in rules.split(";") if p.strip()] if rules else []
    if not parts:
        return "No rule disagreement on this record."
    return " ".join(RULE_PLAIN.get(p, p.replace("_", " ")) for p in parts)


def _build_demo_cases(queue: list[dict], clf: list[dict]) -> list[dict]:
    """Curate a short interactive demo from real tracked outputs."""
    by_id = {row["report_id"]: row for row in queue}
    cases: list[dict] = []

    # Prefer diverse signal patterns for the walkthrough.
    picks: list[tuple[str, str]] = []
    for row in queue:
        if row["heuristic"] and not row["phase1"] and not row["phase2"]:
            picks.append((row["report_id"], "rule_only"))
            break
    for row in queue:
        if row["phase2"] and not row["heuristic"]:
            picks.append((row["report_id"], "llm_only"))
            break
    for row in clf:
        rid = row["report_id"]
        if rid in by_id and by_id[rid]["phase1"]:
            picks.append((rid, "model_flag"))
            break

    seen: set[str] = set()
    for rid, kind in picks:
        if rid in seen or rid not in by_id:
            continue
        seen.add(rid)
        row = by_id[rid]
        clf_row = next((c for c in clf if c["report_id"] == rid), None)
        score = int(row["heuristic"]) + int(row["phase1"]) + int(row["phase2"])
        cases.append(
            {
                "id": rid,
                "kind": kind,
                "entity": row["entity"],
                "reported": row["reported"],
                "summary": (
                    f"Public SGO report {rid}. Reported automation level: "
                    f"{row['reported']}."
                ),
                "rule": {
                    "disagrees": row["heuristic"],
                    "plain": _plain_rules(row["rules"]),
                },
                "model": {
                    "disagrees": row["phase1"],
                    "predicted": row["phase1_pred"]
                    or (clf_row["predicted"] if clf_row else None),
                    "confidence": row["phase1_conf"]
                    if row["phase1_conf"] is not None
                    else (clf_row["confidence"] if clf_row else None),
                    "plain": (
                        "Structured ML model (LightGBM) predicts a different "
                        "automation level from the reported label at high confidence."
                        if row["phase1"]
                        else "Structured ML model agrees with the reported label."
                    ),
                },
                "llm": {
                    "disagrees": row["phase2"],
                    "predicted": row["phase2_pred"],
                    "plain": (
                        "Local LLM narrative reader infers a different automation "
                        "behavior from the crash narrative than the reported label."
                        if row["phase2"]
                        else "No Phase-2 narrative disagreement available/used here."
                    ),
                },
                "verdict_plain": (
                    f"{score} of 3 checks disagree with the reported label → "
                    "queued for human review (not a confirmed error)."
                ),
            }
        )

    # Always include one clean "agreement" teaching case (synthetic illustration).
    cases.append(
        {
            "id": "demo-agree",
            "kind": "agreement",
            "entity": "Illustrative example",
            "reported": "ADS",
            "summary": (
                "Teaching example: an ADS-labeled report whose fields, model "
                "prediction, and narrative all point the same way."
            ),
            "rule": {
                "disagrees": False,
                "plain": "Rule checker finds no hard inconsistency.",
            },
            "model": {
                "disagrees": False,
                "predicted": "ADS",
                "confidence": 0.99,
                "plain": "ML model also predicts ADS — agrees with the label.",
            },
            "llm": {
                "disagrees": False,
                "predicted": "ADS",
                "plain": "Narrative reader also infers ADS behavior.",
            },
            "verdict_plain": (
                "0 of 3 checks disagree → not added to the review queue."
            ),
        }
    )
    return cases


def build_snapshot() -> dict:
    ingest = _load_json("ingest_metadata.json")
    op = _load_json("classifier_operating_point.json")
    metrics = _load_json("classifier_metrics.json")
    consensus = _load_json("consensus_summary.json")
    disagreement = _load_json("disagreement_matrix.json")
    validation = _load_json("validation_report.json")
    queue = [_trim_queue(row) for row in _load_csv("consensus_review_queue.csv")]
    clf = [_trim_clf(row) for row in _load_csv("classifier_flagged_records.csv")]
    heur = _load_csv("heuristic_flags.csv")

    ads = ingest["inputs"]["ads"]["rows_clean"]
    adas = ingest["inputs"]["adas"]["rows_clean"]
    snap = validation["current_data_snapshot"]

    return {
        "meta": {
            "title": "SGO-Audit",
            "tagline": (
                "Finds crash reports whose automation labels do not line up "
                "with the rest of the public record."
            ),
            "data_vintage": ingest.get("data_vintage"),
            "generated_at_utc": consensus.get("generated_at_utc"),
            "responsible_use": consensus.get("responsible_use"),
            "repo_url": "https://github.com/nktarasios/ADAS",
        },
        "corpus": {
            "ads_clean": ads,
            "adas_clean": adas,
            "combined": ads + adas,
        },
        "signals": {
            "heuristic_flags": len(heur),
            "classifier_flags": metrics["flagged_records"]["count"],
            "phase2_sample": disagreement["records_compared"],
            "consensus_queue": consensus["records_in_queue"],
            "phase2_disagrees": consensus["signal_totals"]["phase2_disagrees"],
        },
        "operating_point": {
            "model": op["selected_flag_model"],
            "threshold": op["operating_point"]["threshold"],
            "precision": op["operating_point"]["precision"],
            "recall": op["operating_point"]["recall"],
            "rationale": op["operating_point"]["rationale"],
            "calibration": metrics.get("calibration", "none"),
            "target_precision": 0.80,
        },
        "disagreement": disagreement["counts"],
        "tiers": consensus["tier_counts"],
        "validation": {
            "reference_ids": snap["reference_ids_present_count"]
            + snap["reference_ids_missing_count"],
            "present": snap["reference_ids_present_count"],
            "missing": snap["reference_ids_missing_count"],
            "overlaps": {
                fs["name"]: fs["reference_id_overlap_count"]
                for fs in validation.get("flag_sets", [])
            },
            "caveats": validation.get("caveats", []),
        },
        "entity_counts": Counter(row["entity"] for row in queue).most_common(12),
        "queue": queue,
        "classifier_flags": clf,
        "demo_cases": _build_demo_cases(queue, clf),
        "tools_plain": [
            {
                "id": "rules",
                "name": "Rule checker",
                "does": "Looks for hard contradictions in structured fields and wording.",
                "example": "Labeled ADS, but narrative says the driver was supervising a Level-2 system.",
            },
            {
                "id": "model",
                "name": "ML model (LightGBM)",
                "does": "Predicts ADS vs Level 2 ADAS from structured fields, then flags high-confidence disagreements.",
                "example": "Model is ~99% sure the record looks like ADS, but the file says Level 2 ADAS.",
            },
            {
                "id": "llm",
                "name": "Local LLM narrative reader",
                "does": "Reads the crash narrative and independently infers the automation behavior described.",
                "example": "Narrative sounds like driver-supervised assist, while the label says ADS.",
            },
            {
                "id": "fusion",
                "name": "Consensus queue",
                "does": "Combines the three checks and ranks records where more checks disagree.",
                "example": "2 or 3 disagreements → higher review priority. Still not a verdict.",
            },
        ],
    }


def main() -> None:
    snapshot = build_snapshot()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "window.SGO_SHOWCASE = "
        + json.dumps(snapshot, indent=2, ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )
    try:
        display = OUT.relative_to(ROOT)
    except ValueError:
        display = OUT
    print(f"Wrote {display} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
