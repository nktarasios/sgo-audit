"""Tests for the report-version corrections answer key."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.corrections import (
    AUTOMATION_LEVEL_KIND,
    CORRECTION_COLUMNS,
    OTHER_KIND,
    build_corrections_file,
    canonical_automation_label,
    extract_corrections,
    format_summary_lines,
    load_versioned_reports,
    main,
    summarize_corrections_file,
)
from src.fields import (
    ADAS_LEVEL,
    ADS_LEVEL,
    AUTOMATION_SYSTEM_ENGAGED,
    DRIVER_OPERATOR_TYPE,
    REPORT_ID,
    REPORT_SUBMISSION_DATE,
    REPORT_VERSION,
    REPORTED_AUTOMATION_LEVEL,
    ROADWAY_TYPE,
    SOURCE_FILE,
)
from src.ingest import keep_latest_report_versions

FIXTURE_DIR = Path(__file__).parent / "fixtures"
ADS_FIXTURE = FIXTURE_DIR / "corrections_ads.csv"
ADAS_FIXTURE = FIXTURE_DIR / "corrections_adas.csv"


def test_canonical_automation_label_maps_archive_values():
    assert canonical_automation_label("ADS") == ADS_LEVEL
    assert canonical_automation_label(" adas ") == ADAS_LEVEL
    assert canonical_automation_label("Unknown, see Narrative") == "Unknown"
    assert canonical_automation_label("Unknown, see Narrative.") == "Unknown"
    assert canonical_automation_label("Level 2 ADAS") == ADAS_LEVEL
    assert canonical_automation_label("") is None
    assert canonical_automation_label(pd.NA) is None


def test_fixture_records_level_changes_before_latest_version_collapse():
    versioned = load_versioned_reports(ADS_FIXTURE, ADAS_FIXTURE)
    corrections, summary = extract_corrections(versioned)

    assert list(corrections.columns) == list(CORRECTION_COLUMNS)
    assert summary["label_source"] == AUTOMATION_SYSTEM_ENGAGED
    assert summary["automation_level_corrections"] == 3
    assert summary["automation_level_rows"] == 4
    assert summary["other_field_changes"] == 3
    assert summary["other_field_rows"] == 3
    assert set(corrections[REPORT_ID]) == {
        "R-BLANK",
        "R-FLIP",
        "R-LEVEL",
        "R-OTHER",
        "R-PERIOD",
        "R-UNK",
    }

    by_key = {
        (row[REPORT_ID], row["versions"]): row
        for row in corrections.to_dict(orient="records")
    }

    level = by_key[("R-LEVEL", "1->2")]
    assert level["correction_kind"] == AUTOMATION_LEVEL_KIND
    assert level["old_label"] == ADAS_LEVEL
    assert level["new_label"] == ADS_LEVEL
    assert level["dates"] == "JAN-2022->FEB-2022"
    assert level["old_operator_type"] == "Consumer"
    assert level["new_operator_type"] == "None"
    assert level["context_field_changes"] == (
        "Driver / Operator Type: Consumer -> None"
        " | Roadway Type: Street -> Highway"
        " | SV Pre-Crash Movement: Stopped -> Going Straight"
        " | SV Precrash Speed (MPH): 0 -> 25"
    )

    first_flip = by_key[("R-FLIP", "1->2")]
    second_flip = by_key[("R-FLIP", "2->3")]
    assert first_flip["old_label"] == ADS_LEVEL
    assert first_flip["new_label"] == ADAS_LEVEL
    assert first_flip["context_field_changes"] == ""
    assert second_flip["old_label"] == ADAS_LEVEL
    assert second_flip["new_label"] == ADS_LEVEL
    assert second_flip["context_field_changes"] == "Roadway Type: Street -> Highway"
    assert second_flip["dates"] == "FEB-2022->MAR-2022"

    blank = by_key[("R-BLANK", "1->2")]
    assert blank["correction_kind"] == OTHER_KIND
    assert blank["old_label"] == "(blank)"
    assert blank["new_label"] == ADAS_LEVEL

    unknown = by_key[("R-UNK", "1->2")]
    assert unknown["old_label"] == "Unknown"
    assert unknown["new_label"] == ADS_LEVEL
    assert unknown["correction_kind"] == OTHER_KIND

    other = by_key[("R-OTHER", "1->2")]
    assert other["old_label"] == ADAS_LEVEL
    assert other["new_label"] == "Unknown"
    assert other["source_file"] == "corrections_adas.csv"

    period = by_key[("R-PERIOD", "1->2")]
    assert period["dates"] == "2023-03->2023-04"
    assert period["correction_kind"] == AUTOMATION_LEVEL_KIND

    collapsed, collapsed_summary = extract_corrections(
        keep_latest_report_versions(versioned)
    )
    assert collapsed.empty
    assert collapsed_summary["correction_rows"] == 0
    assert collapsed_summary["automation_level_corrections"] == 0


def test_source_file_label_used_when_engagement_column_is_not_a_level():
    frame = pd.DataFrame(
        {
            REPORT_ID: ["X", "X"],
            REPORT_VERSION: ["1", "2"],
            AUTOMATION_SYSTEM_ENGAGED: ["Verified Engaged", "Verified Engaged"],
            REPORTED_AUTOMATION_LEVEL: [ADAS_LEVEL, ADS_LEVEL],
            SOURCE_FILE: ["adas.csv", "ads.csv"],
            DRIVER_OPERATOR_TYPE: ["Consumer", "None"],
            ROADWAY_TYPE: ["Street", "Street"],
            REPORT_SUBMISSION_DATE: ["JAN-2024", "MAR-2024"],
        }
    )

    corrections, summary = extract_corrections(frame)

    assert summary["label_source"] == REPORTED_AUTOMATION_LEVEL
    assert len(corrections) == 1
    row = corrections.iloc[0]
    assert row["correction_kind"] == AUTOMATION_LEVEL_KIND
    assert row["old_label"] == ADAS_LEVEL
    assert row["new_label"] == ADS_LEVEL
    assert row["versions"] == "1->2"
    assert row["dates"] == "JAN-2024->MAR-2024"
    assert row["old_operator_type"] == "Consumer"
    assert row["new_operator_type"] == "None"
    assert "Driver / Operator Type: Consumer -> None" in row["context_field_changes"]
    assert "source_file: adas.csv -> ads.csv" in row["context_field_changes"]


def test_cli_writes_corrections_csv(tmp_path):
    output = tmp_path / "corrections.csv"
    code = main(
        [
            "--ads-path",
            str(ADS_FIXTURE),
            "--adas-path",
            str(ADAS_FIXTURE),
            "--output",
            str(output),
        ]
    )

    assert code == 0
    assert output.exists()
    summary = summarize_corrections_file(output)
    assert summary["automation_level_corrections"] == 3
    assert summary["correction_rows"] == 7
    lines = format_summary_lines(summary)
    assert any("`3` reports (`4` rows)" in line for line in lines)

    written = build_corrections_file(
        ads_path=ADS_FIXTURE,
        adas_path=ADAS_FIXTURE,
        output_path=tmp_path / "again.csv",
    )
    assert written["output"].endswith("again.csv")
    assert Path(written["output"]).exists()
