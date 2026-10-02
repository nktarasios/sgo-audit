from pathlib import Path

import pandas as pd

from src.fields import (
    ADAS_LEVEL,
    ADS_LEVEL,
    NARRATIVE,
    REPORT_ID,
    REPORT_PERIOD,
    REPORTED_AUTOMATION_LEVEL,
    SOURCE_FILE,
    redaction_flag_column,
)
from src.ingest import ingest_files, is_redaction_marker


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_redaction_marker_detection():
    assert is_redaction_marker("[REDACTED, MAY CONTAIN CONFIDENTIAL BUSINESS INFORMATION]")
    assert is_redaction_marker("[MAY CONTAIN PERSONALLY IDENTIFIABLE INFORMATION]")
    assert is_redaction_marker("[XXX]")
    assert not is_redaction_marker("Consumer driver-assistance crash narrative")
    assert not is_redaction_marker("")


def test_ingest_files_cleans_redactions_and_keeps_latest_versions(tmp_path):
    result = ingest_files(
        ads_path=FIXTURE_DIR / "ads_fixture.csv",
        adas_path=FIXTURE_DIR / "adas_fixture.csv",
        processed_dir=tmp_path,
        vintage_path=FIXTURE_DIR / "DATA_VINTAGE.txt",
    )

    assert len(result.ads) == 2
    assert len(result.adas) == 2
    assert len(result.combined) == 4
    assert result.metadata["data_vintage"] == "synthetic-fixture"

    latest_ads_1 = result.ads.loc[result.ads[REPORT_ID] == "ADS-1"].iloc[0]
    assert latest_ads_1[REPORTED_AUTOMATION_LEVEL] == ADS_LEVEL
    assert latest_ads_1[SOURCE_FILE] == "ads_fixture.csv"
    assert latest_ads_1[REPORT_PERIOD] == "2024-01"
    assert pd.isna(latest_ads_1[NARRATIVE])
    assert bool(latest_ads_1[redaction_flag_column(NARRATIVE)])

    adas_2 = result.adas.loc[result.adas[REPORT_ID] == "ADAS-2"].iloc[0]
    assert adas_2[REPORTED_AUTOMATION_LEVEL] == ADAS_LEVEL
    assert pd.isna(adas_2[NARRATIVE])
    assert bool(adas_2[redaction_flag_column(NARRATIVE)])

    assert (tmp_path / "ads_clean.csv").exists()
    assert (tmp_path / "adas_clean.csv").exists()
    assert (tmp_path / "combined_audit_frame.csv").exists()
    assert (tmp_path / "ingest_metadata.json").exists()
