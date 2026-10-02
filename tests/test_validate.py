import json

import pandas as pd

from src.fields import REPORT_ID
from src.validate import (
    HIGH_PRIORITY_COMPARISON_CATEGORY,
    build_validation_report,
    extract_reference_ids,
    run_validation,
)


def test_extract_reference_ids_deduplicates_sources_and_object_entries():
    reference = {
        "investigations": [
            {
                "investigation_name": "Investigation A",
                "url": "https://example.test/a.pdf",
                "notes": "First source.",
                "report_ids": ["R-1", {"report_id": "R-2", "notes": "Per-ID note."}],
            },
            {
                "investigation_name": "Investigation B",
                "url": "https://example.test/b.pdf",
                "notes": "Second source.",
                "report_ids": ["R-2", " "],
            },
        ]
    }

    ids, sources_by_id = extract_reference_ids(reference)

    assert ids == {"R-1", "R-2"}
    assert len(sources_by_id["R-2"]) == 2
    assert sources_by_id["R-2"][0]["report_id_notes"] == "Per-ID note."


def write_reference(path):
    reference = {
        "description": "Synthetic reference set.",
        "archive_2021_2025_expected_present_ids": ["R-1", "R-2"],
        "archive_2021_2025_expected_missing_ids": ["R-3"],
        "investigations": [
            {
                "investigation_name": "Synthetic ODI Resume",
                "url": "https://example.test/resume.pdf",
                "notes": "Tiny fixture for tests.",
                "report_ids": ["R-1", "R-2", "R-3"],
            }
        ],
    }
    path.write_text(json.dumps(reference), encoding="utf-8")


def test_build_validation_report_counts_snapshot_presence_and_flag_overlap(tmp_path):
    reference_path = tmp_path / "reference.json"
    combined_path = tmp_path / "combined.csv"
    heuristic_path = tmp_path / "heuristic.csv"
    classifier_path = tmp_path / "classifier.csv"
    comparison_path = tmp_path / "comparison.csv"
    write_reference(reference_path)
    pd.DataFrame([{REPORT_ID: "R-1"}, {REPORT_ID: "R-2"}, {REPORT_ID: "OTHER"}]).to_csv(
        combined_path, index=False
    )
    pd.DataFrame([{REPORT_ID: "R-2"}, {REPORT_ID: "OTHER"}]).to_csv(heuristic_path, index=False)
    pd.DataFrame([{REPORT_ID: "R-3"}]).to_csv(classifier_path, index=False)
    pd.DataFrame(
        [
            {REPORT_ID: "R-1", "disagreement_category": HIGH_PRIORITY_COMPARISON_CATEGORY},
            {REPORT_ID: "R-2", "disagreement_category": "all_agree"},
            {REPORT_ID: "R-3", "disagreement_category": HIGH_PRIORITY_COMPARISON_CATEGORY},
        ]
    ).to_csv(comparison_path, index=False)

    report = build_validation_report(
        reference_path=reference_path,
        combined_frame_path=combined_path,
        heuristic_flags_path=heuristic_path,
        classifier_flags_path=classifier_path,
        phase_comparison_path=comparison_path,
        ingest_metadata_path=tmp_path / "missing_metadata.json",
    )

    snapshot = report["current_data_snapshot"]
    assert snapshot["reference_ids_present_count"] == 2
    assert snapshot["reference_ids_present"] == ["R-1", "R-2"]
    assert snapshot["reference_ids_missing"] == ["R-3"]
    assert snapshot["expected_present_ids_not_found"] == []
    assert snapshot["expected_missing_ids_found"] == []

    flag_sets = {flag_set["name"]: flag_set for flag_set in report["flag_sets"]}
    assert flag_sets["heuristic_flags"]["snapshot_present_reference_id_overlap_ids"] == ["R-2"]
    assert flag_sets["classifier_flagged_records"]["reference_id_overlap_ids"] == ["R-3"]
    assert flag_sets["classifier_flagged_records"]["snapshot_present_reference_id_overlap_count"] == 0
    assert flag_sets["phase1_phase2_agree_vs_reported"]["records_loaded"] == 2
    assert flag_sets["phase1_phase2_agree_vs_reported"]["reference_id_overlap_ids"] == ["R-1", "R-3"]
    assert flag_sets["phase1_phase2_agree_vs_reported"]["snapshot_present_reference_id_overlap_ids"] == ["R-1"]
    assert "not labeled ground truth" in report["validation_framing"]


def test_run_validation_writes_report_and_note_with_missing_optional_flags(tmp_path):
    reference_path = tmp_path / "reference.json"
    combined_path = tmp_path / "combined.csv"
    report_path = tmp_path / "validation_report.json"
    note_path = tmp_path / "validation_note.md"
    write_reference(reference_path)
    pd.DataFrame([{REPORT_ID: "R-1"}, {REPORT_ID: "R-2"}]).to_csv(combined_path, index=False)

    report = run_validation(
        reference_path=reference_path,
        combined_frame_path=combined_path,
        heuristic_flags_path=tmp_path / "missing_heuristic.csv",
        classifier_flags_path=tmp_path / "missing_classifier.csv",
        phase_comparison_path=tmp_path / "missing_comparison.csv",
        ingest_metadata_path=tmp_path / "missing_metadata.json",
        report_path=report_path,
        note_path=note_path,
    )

    assert report_path.exists()
    assert note_path.exists()
    written_report = json.loads(report_path.read_text(encoding="utf-8"))
    note = note_path.read_text(encoding="utf-8")
    assert written_report["current_data_snapshot"]["reference_ids_present_count"] == 2
    assert {flag_set["status"] for flag_set in written_report["flag_sets"]} == {"missing"}
    assert "Flags are not findings" in note
    assert "small validation sample" in note
    assert report["validation_report_path"] == str(report_path)
