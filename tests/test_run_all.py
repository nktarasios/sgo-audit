"""Regression tests for scripts/run_all.py publish helpers."""

from __future__ import annotations

import json

import pytest

from scripts.run_all import build_summary, count_reference_report_ids


def test_build_summary_handles_object_shaped_phase3_report_ids(tmp_path, monkeypatch):
    """Object-shaped Phase 3 report_ids must not crash publish (PR #4 fix).

    The curated ``nhtsa_investigation_report_ids.json`` mixes plain string IDs
    with ``{"report_id": "...", "notes": "..."}`` objects. The pre-fix path did
    ``set.update(report_ids)`` and raised ``TypeError: unhashable type: 'dict'``,
    aborting ``build_summary`` before ``results/`` could be written.

    This synthetic fixture also asserts that the run-level ``data_vintage`` from
    ingest metadata is surfaced in ``RUN_SUMMARY.md`` (TECHNICAL_IMPLEMENTATION 3.1).
    """

    import scripts.run_all as run_all

    reference = {
        "investigations": [
            {
                "investigation_name": "EA-synthetic",
                "report_ids": [
                    {"report_id": "13781-11937", "notes": "object-shaped"},
                    {"report_id": "13781-8004", "notes": "also object"},
                ],
            },
            {
                "investigation_name": "PE-synthetic",
                "report_ids": ["13781-8004", "13781-7181", " "],
            },
        ]
    }

    # Reproduce the original crash mode against the same fixture shape.
    with pytest.raises(TypeError, match="unhashable type"):
        broken: set = set()
        for inv in reference["investigations"]:
            broken.update(inv.get("report_ids", []))

    assert count_reference_report_ids(reference) == 3

    root = tmp_path
    processed = root / "data" / "processed"
    reference_dir = root / "data" / "reference"
    raw = root / "data" / "raw"
    processed.mkdir(parents=True)
    reference_dir.mkdir(parents=True)
    raw.mkdir(parents=True)

    (processed / "ingest_metadata.json").write_text(
        json.dumps(
            {
                "data_vintage": "Archive-2021-2025",
                "inputs": {
                    "ads": {"rows_clean": 10},
                    "adas": {"rows_clean": 20},
                },
            }
        ),
        encoding="utf-8",
    )
    (processed / "classifier_operating_point.json").write_text(
        json.dumps(
            {
                "selected_flag_model": "lightgbm",
                "operating_point": {
                    "threshold": 0.59,
                    "precision": 0.912,
                    "recall": 0.378,
                    "rationale": "synthetic fixture",
                },
            }
        ),
        encoding="utf-8",
    )
    (processed / "classifier_metrics.json").write_text(
        json.dumps({"calibration": "none", "selected_flag_model": "lightgbm"}),
        encoding="utf-8",
    )
    (processed / "disagreement_matrix.json").write_text(
        json.dumps({"counts": {"all_agree": 1}}),
        encoding="utf-8",
    )
    (processed / "consensus_summary.json").write_text(
        json.dumps(
            {
                "records_in_queue": 0,
                "tier_counts": {},
                "records_with_phase2_evidence": 0,
            }
        ),
        encoding="utf-8",
    )
    (processed / "validation_report.json").write_text(
        json.dumps(
            {
                "current_data_snapshot": {
                    "reference_ids_present_count": 1,
                    "reference_ids_missing": ["13781-11937"],
                },
                "flag_sets": [],
                "reference": {"investigations": []},
            }
        ),
        encoding="utf-8",
    )
    (reference_dir / "nhtsa_investigation_report_ids.json").write_text(
        json.dumps(reference),
        encoding="utf-8",
    )
    (raw / "DATA_VINTAGE.txt").write_text(
        "DATA_VINTAGE=Archive-2021-2025\nDownloaded from test\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(run_all, "ROOT", root)

    summary = build_summary(
        vintage="should-be-overridden-by-ingest",
        llm_backend="transformers",
        llm_sample=5,
        ollama_model="Qwen/Qwen2.5-1.5B-Instruct",
    )

    assert "Data vintage: **Archive-2021-2025**" in summary
    assert "Reference IDs: `3`" in summary
    assert "DATA_VINTAGE=" not in summary
