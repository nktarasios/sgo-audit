import pandas as pd

from src.compare import build_comparison, build_matrix, categorize_row
from src.fields import ADAS_LEVEL, ADS_LEVEL, REPORT_ID, REPORTED_AUTOMATION_LEVEL, REPORTING_ENTITY


def test_categorize_row_requested_matrix_categories():
    assert (
        categorize_row(
            pd.Series(
                {
                    REPORTED_AUTOMATION_LEVEL: ADS_LEVEL,
                    "phase1_predicted_automation_level": ADS_LEVEL,
                    "phase2_llm_inferred_automation_level": ADS_LEVEL,
                }
            )
        )
        == "all_agree"
    )
    assert (
        categorize_row(
            pd.Series(
                {
                    REPORTED_AUTOMATION_LEVEL: ADAS_LEVEL,
                    "phase1_predicted_automation_level": ADS_LEVEL,
                    "phase2_llm_inferred_automation_level": ADS_LEVEL,
                }
            )
        )
        == "phase1_phase2_agree_vs_reported"
    )
    assert (
        categorize_row(
            pd.Series(
                {
                    REPORTED_AUTOMATION_LEVEL: ADS_LEVEL,
                    "phase1_predicted_automation_level": ADS_LEVEL,
                    "phase2_llm_inferred_automation_level": ADAS_LEVEL,
                }
            )
        )
        == "phase1_phase2_disagree"
    )
    assert (
        categorize_row(
            pd.Series(
                {
                    REPORTED_AUTOMATION_LEVEL: ADS_LEVEL,
                    "phase1_predicted_automation_level": "",
                    "phase2_llm_inferred_automation_level": ADAS_LEVEL,
                }
            )
        )
        == "only_one_method_available"
    )


def test_build_comparison_and_matrix_with_synthetic_rows():
    phase1 = pd.DataFrame(
        [
            {
                REPORT_ID: "R-1",
                REPORTING_ENTITY: "Entity A",
                REPORTED_AUTOMATION_LEVEL: ADS_LEVEL,
                "phase1_predicted_automation_level": ADS_LEVEL,
                "phase1_prediction_confidence": "0.91",
            },
            {
                REPORT_ID: "R-2",
                REPORTING_ENTITY: "Entity B",
                REPORTED_AUTOMATION_LEVEL: ADAS_LEVEL,
                "phase1_predicted_automation_level": ADS_LEVEL,
                "phase1_prediction_confidence": "0.92",
            },
            {
                REPORT_ID: "R-3",
                REPORTING_ENTITY: "Entity C",
                REPORTED_AUTOMATION_LEVEL: ADS_LEVEL,
                "phase1_predicted_automation_level": ADS_LEVEL,
                "phase1_prediction_confidence": "0.93",
            },
        ]
    )
    phase2 = pd.DataFrame(
        [
            {
                REPORT_ID: "R-1",
                REPORTING_ENTITY: "Entity A",
                REPORTED_AUTOMATION_LEVEL: ADS_LEVEL,
                "phase2_llm_inferred_automation_level": ADS_LEVEL,
                "phase2_inference_backend": "ollama",
                "phase2_inference_status": "ok",
                "phase2_llm_rationale": "mentions Level 4 ADS",
            },
            {
                REPORT_ID: "R-2",
                REPORTING_ENTITY: "Entity B",
                REPORTED_AUTOMATION_LEVEL: ADAS_LEVEL,
                "phase2_llm_inferred_automation_level": ADS_LEVEL,
                "phase2_inference_backend": "ollama",
                "phase2_inference_status": "ok",
                "phase2_llm_rationale": "mentions autonomous mode",
            },
            {
                REPORT_ID: "R-3",
                REPORTING_ENTITY: "Entity C",
                REPORTED_AUTOMATION_LEVEL: ADS_LEVEL,
                "phase2_llm_inferred_automation_level": ADAS_LEVEL,
                "phase2_inference_backend": "ollama",
                "phase2_inference_status": "ok",
                "phase2_llm_rationale": "mentions driver took over",
            },
            {
                REPORT_ID: "R-4",
                REPORTING_ENTITY: "Entity D",
                REPORTED_AUTOMATION_LEVEL: ADAS_LEVEL,
                "phase2_llm_inferred_automation_level": ADAS_LEVEL,
                "phase2_inference_backend": "ollama",
                "phase2_inference_status": "ok",
                "phase2_llm_rationale": "mentions Level 2",
            },
        ]
    )

    comparison = build_comparison(phase1, phase2)
    matrix = build_matrix(comparison)

    by_id = comparison.set_index(REPORT_ID)["disagreement_category"].to_dict()
    assert by_id["R-1"] == "all_agree"
    assert by_id["R-2"] == "phase1_phase2_agree_vs_reported"
    assert by_id["R-3"] == "phase1_phase2_disagree"
    assert by_id["R-4"] == "only_one_method_available"
    assert matrix["counts"] == {
        "all_agree": 1,
        "phase1_phase2_agree_vs_reported": 1,
        "phase1_phase2_disagree": 1,
        "only_one_method_available": 1,
    }
    assert "human review" in matrix["responsible_use"]
