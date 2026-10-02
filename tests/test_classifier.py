import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.classifier import (
    HEURISTIC_BOOLEAN_FEATURES,
    LEAKAGE_COLUMNS,
    build_feature_matrix,
    build_feature_spec,
    build_flagged_records,
    build_prediction_records,
    calibrate_model_step,
    make_logistic_pipeline,
    sweep_flag_thresholds,
)
from src.fields import (
    ADAS_LEVEL,
    ADS_EQUIPPED,
    ADS_LEVEL,
    AUTOMATION_SYSTEM_ENGAGED,
    CRASH_WITH,
    DRIVER_OPERATOR_TYPE,
    HIGHEST_INJURY_SEVERITY,
    NARRATIVE,
    REPORT_ID,
    REPORTED_AUTOMATION_LEVEL,
    REPORTING_ENTITY,
    ROADWAY_TYPE,
    SOURCE_FILE,
    SV_PRE_CRASH_MOVEMENT,
    SV_PRECRASH_SPEED,
    WITHIN_ODD,
)


def synthetic_classifier_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                REPORT_ID: "R-1",
                REPORTING_ENTITY: "Example ADS",
                REPORTED_AUTOMATION_LEVEL: ADS_LEVEL,
                SOURCE_FILE: "ads.csv",
                AUTOMATION_SYSTEM_ENGAGED: "ADS",
                NARRATIVE: "Narrative should be excluded.",
                ADS_EQUIPPED: "Yes",
                DRIVER_OPERATOR_TYPE: "None",
                ROADWAY_TYPE: "Street",
                CRASH_WITH: "Passenger Car",
                HIGHEST_INJURY_SEVERITY: "No Injuries Reported",
                SV_PRE_CRASH_MOVEMENT: "Stopped",
                SV_PRECRASH_SPEED: "0",
                WITHIN_ODD: "Yes",
                "Lighting": "Daylight",
                "Weather - Clear": "Y",
                "ads_file_but_adas_engaged": False,
                "adas_file_but_ads_engaged": False,
                "ads_label_but_consumer_driver": False,
                "adas_label_but_no_driver": False,
                "ads_narrative_level2_language": False,
                "any_heuristic_flag": False,
                "rules_triggered": "",
                "reasons": "",
            },
            {
                REPORT_ID: "R-2",
                REPORTING_ENTITY: "Example ADAS",
                REPORTED_AUTOMATION_LEVEL: ADAS_LEVEL,
                SOURCE_FILE: "adas.csv",
                AUTOMATION_SYSTEM_ENGAGED: "ADAS",
                NARRATIVE: "Narrative should be excluded.",
                ADS_EQUIPPED: "No",
                DRIVER_OPERATOR_TYPE: "Consumer",
                ROADWAY_TYPE: "Highway",
                CRASH_WITH: "Fixed Object",
                HIGHEST_INJURY_SEVERITY: "Minor Injury",
                SV_PRE_CRASH_MOVEMENT: "Proceeding Straight",
                SV_PRECRASH_SPEED: "45",
                WITHIN_ODD: "No",
                "Lighting": "Dark - Lighted",
                "Weather - Clear": "",
                "ads_file_but_adas_engaged": False,
                "adas_file_but_ads_engaged": True,
                "ads_label_but_consumer_driver": False,
                "adas_label_but_no_driver": False,
                "ads_narrative_level2_language": False,
                "any_heuristic_flag": True,
                "rules_triggered": "adas_file_but_ads_engaged",
                "reasons": "ADAS source file reports Automation System Engaged? as ADS",
            },
        ]
    )


def test_feature_matrix_excludes_known_leakage_columns():
    frame = synthetic_classifier_frame()

    matrix, spec = build_feature_matrix(frame)

    assert set(matrix.columns).isdisjoint(LEAKAGE_COLUMNS)
    assert REPORT_ID not in matrix.columns
    assert NARRATIVE not in matrix.columns
    assert ADS_EQUIPPED not in matrix.columns
    assert SOURCE_FILE not in matrix.columns
    assert REPORTED_AUTOMATION_LEVEL not in matrix.columns
    assert AUTOMATION_SYSTEM_ENGAGED not in matrix.columns
    assert DRIVER_OPERATOR_TYPE in matrix.columns
    assert SV_PRECRASH_SPEED in matrix.columns
    assert "adas_file_but_ads_engaged" in matrix.columns
    assert set(spec.heuristic_boolean).issubset(set(HEURISTIC_BOOLEAN_FEATURES))


def test_build_feature_spec_documents_ads_equipped_exclusion():
    frame = synthetic_classifier_frame()

    spec = build_feature_spec(frame)

    assert ADS_EQUIPPED in spec.excluded_high_risk
    assert ADS_EQUIPPED not in spec.selected_columns


def test_disagreement_flagging_respects_threshold_and_ranking():
    frame = synthetic_classifier_frame()
    predictions = pd.DataFrame(
        {
            "predicted_automation_level": [ADAS_LEVEL, ADS_LEVEL],
            "prediction_confidence": [0.91, 0.72],
        }
    )

    flagged = build_flagged_records(frame, predictions, threshold=0.80, model_name="unit-test-model")

    assert flagged[REPORT_ID].tolist() == ["R-1"]
    assert flagged["predicted_automation_level"].tolist() == [ADAS_LEVEL]
    assert flagged["disagreement_confidence"].tolist() == [0.91]


def test_build_prediction_records_includes_all_rows_for_phase2_compare():
    frame = synthetic_classifier_frame()
    predictions = pd.DataFrame(
        {
            "predicted_automation_level": [ADAS_LEVEL, ADS_LEVEL],
            "prediction_confidence": [0.91, 0.72],
        }
    )

    records = build_prediction_records(frame, predictions, threshold=0.80, model_name="unit-test-model")

    assert records[REPORT_ID].tolist() == ["R-1", "R-2"]
    assert records["predicted_automation_level"].tolist() == [ADAS_LEVEL, ADS_LEVEL]
    assert records["classifier_flag_for_review"].tolist() == [True, False]
    assert "responsible_use" in records.columns


def test_calibrate_model_step_none_is_passthrough():
    frame = synthetic_classifier_frame()
    spec = build_feature_spec(frame)
    base = make_logistic_pipeline(spec)

    assert calibrate_model_step(base, method="none") is base


def test_calibrate_model_step_wraps_and_predicts_probabilities():
    rows = []
    for i in range(20):
        source = synthetic_classifier_frame().iloc[i % 2].to_dict()
        source[REPORT_ID] = f"R-{i}"
        source[SV_PRECRASH_SPEED] = str((i * 3) % 60)
        rows.append(source)
    frame = pd.DataFrame(rows)

    spec = build_feature_spec(frame)
    features, _ = build_feature_matrix(frame, spec)
    labels = frame[REPORTED_AUTOMATION_LEVEL]

    calibrated = calibrate_model_step(make_logistic_pipeline(spec), method="sigmoid", cv=3)
    calibrated.fit(features, labels)

    model_step = calibrated.named_steps["model"]
    assert isinstance(model_step, CalibratedClassifierCV)
    assert hasattr(model_step, "classes_")

    probabilities = calibrated.predict_proba(features)
    assert probabilities.shape == (len(frame), 2)
    assert probabilities.sum(axis=1).round(6).tolist() == [1.0] * len(frame)


def test_threshold_sweep_counts_and_recall_are_monotonic_with_threshold():
    sweep = sweep_flag_thresholds(
        reported_labels=[ADS_LEVEL, ADS_LEVEL, ADAS_LEVEL, ADAS_LEVEL],
        predicted_labels=[ADAS_LEVEL, ADS_LEVEL, ADS_LEVEL, ADS_LEVEL],
        confidences=[0.95, 0.90, 0.85, 0.60],
        truth_is_misclassified=[True, False, False, True],
        thresholds=[0.50, 0.80, 0.90, 0.96],
    )

    assert sweep["flagged_count"].tolist() == [3, 2, 1, 0]
    assert sweep["recall"].tolist() == [1.0, 0.5, 0.5, 0.0]
    assert sweep["flagged_count"].is_monotonic_decreasing
    assert sweep["recall"].is_monotonic_decreasing
    assert sweep.loc[sweep["threshold"] == 0.90, "precision"].iloc[0] == 1.0
