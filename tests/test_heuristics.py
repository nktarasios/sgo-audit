import pandas as pd

from src.fields import (
    ADAS_LEVEL,
    ADS_LEVEL,
    AUTOMATION_SYSTEM_ENGAGED,
    DRIVER_OPERATOR_TYPE,
    NARRATIVE,
    REPORT_ID,
    REPORT_MONTH,
    REPORT_YEAR,
    REPORTED_AUTOMATION_LEVEL,
    REPORTING_ENTITY,
    SOURCE_FILE,
)
from src.heuristics import (
    adas_file_but_ads_engaged,
    adas_label_but_no_driver,
    ads_file_but_adas_engaged,
    ads_label_but_consumer_driver,
    ads_narrative_level2_language,
    flags_for_review,
    score_records,
)


def base_record(**overrides):
    record = {
        REPORT_ID: "R-1",
        REPORTING_ENTITY: "Example Entity",
        REPORT_YEAR: "2024",
        REPORT_MONTH: "1",
        SOURCE_FILE: "source.csv",
        REPORTED_AUTOMATION_LEVEL: ADS_LEVEL,
        AUTOMATION_SYSTEM_ENGAGED: "ADS",
        DRIVER_OPERATOR_TYPE: "In-Vehicle (Commercial/Test)",
        NARRATIVE: "Routine report narrative.",
    }
    record.update(overrides)
    return record


def test_ads_file_but_adas_engaged_fires_and_does_not_fire():
    flag, reason = ads_file_but_adas_engaged(
        base_record(**{AUTOMATION_SYSTEM_ENGAGED: "ADAS"})
    )
    assert flag
    assert "ADS source file" in reason

    flag, reason = ads_file_but_adas_engaged(
        base_record(**{AUTOMATION_SYSTEM_ENGAGED: "ADS"})
    )
    assert not flag
    assert reason == ""


def test_adas_file_but_ads_engaged_fires_and_does_not_fire():
    flag, reason = adas_file_but_ads_engaged(
        base_record(
            **{
                REPORTED_AUTOMATION_LEVEL: ADAS_LEVEL,
                AUTOMATION_SYSTEM_ENGAGED: "ADS",
            }
        )
    )
    assert flag
    assert "ADAS source file" in reason

    flag, reason = adas_file_but_ads_engaged(
        base_record(
            **{
                REPORTED_AUTOMATION_LEVEL: ADAS_LEVEL,
                AUTOMATION_SYSTEM_ENGAGED: "ADAS",
            }
        )
    )
    assert not flag
    assert reason == ""


def test_ads_label_but_consumer_driver_fires_and_does_not_fire():
    flag, reason = ads_label_but_consumer_driver(
        base_record(**{DRIVER_OPERATOR_TYPE: "Consumer"})
    )
    assert flag
    assert "Consumer" in reason

    flag, reason = ads_label_but_consumer_driver(
        base_record(**{DRIVER_OPERATOR_TYPE: "Remote"})
    )
    assert not flag
    assert reason == ""


def test_adas_label_but_no_driver_fires_and_does_not_fire():
    flag, reason = adas_label_but_no_driver(
        base_record(
            **{
                REPORTED_AUTOMATION_LEVEL: ADAS_LEVEL,
                DRIVER_OPERATOR_TYPE: "None",
            }
        )
    )
    assert flag
    assert "None" in reason

    flag, reason = adas_label_but_no_driver(
        base_record(
            **{
                REPORTED_AUTOMATION_LEVEL: ADAS_LEVEL,
                DRIVER_OPERATOR_TYPE: "Consumer",
            }
        )
    )
    assert not flag
    assert reason == ""


def test_ads_narrative_level2_language_fires_narrowly():
    flag, reason = ads_narrative_level2_language(
        base_record(**{NARRATIVE: "The driver took over after Autopilot warnings."})
    )
    assert flag
    assert "Level-2" in reason

    flag, reason = ads_narrative_level2_language(
        base_record(
            **{
                REPORTED_AUTOMATION_LEVEL: ADAS_LEVEL,
                NARRATIVE: "The driver took over after Autopilot warnings.",
            }
        )
    )
    assert not flag
    assert reason == ""


def test_score_records_and_flags_for_review_compact_output():
    frame = pd.DataFrame(
        [
            base_record(**{REPORT_ID: "R-1", AUTOMATION_SYSTEM_ENGAGED: "ADAS"}),
            base_record(**{REPORT_ID: "R-2"}),
        ]
    )

    scored = score_records(frame)
    flags = flags_for_review(scored)

    assert int(scored["any_heuristic_flag"].sum()) == 1
    assert flags[REPORT_ID].tolist() == ["R-1"]
    assert flags["rules_triggered"].iloc[0] == "ads_file_but_adas_engaged"
