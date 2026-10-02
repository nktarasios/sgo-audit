"""Phase 1 classical classifier for SGO automation-level audit signals.

The model predicts the reported ADS vs. Level 2 ADAS label from structured
fields and Phase 0 heuristic booleans, then ranks records where a high
confidence prediction disagrees with the reported label. Outputs are review
signals only, not findings of misclassification or safety-rate claims.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .fields import (
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
from .heuristics import RULES, score_records

DEFAULT_INPUT_PATH = Path("data/processed/heuristic_scored_records.csv")
DEFAULT_COMBINED_INPUT_PATH = Path("data/processed/combined_audit_frame.csv")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_OUTPUTS_DIR = Path("outputs")
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.25
DEFAULT_SYNTHETIC_FLIP_FRACTION = 0.10
# Precision-first Phase 1 operating point: require ≥80% synthetic-mislabel
# precision, then maximize recall within that constraint (see choose_operating_point).
DEFAULT_TARGET_SYNTHETIC_PRECISION = 0.80
# Default stays "none": on Archive-2021-2025, sigmoid/isotonic did not raise
# precision at the chosen operating point and expanded the flag set at lower
# mean confidence — opposite of precision-first Phase 1 flagging. Keep
# --calibration {sigmoid,isotonic} as an opt-in exploration switch.
DEFAULT_CALIBRATION = "none"
CALIBRATION_CHOICES = ("none", "sigmoid", "isotonic")
CALIBRATION_CV = 3

LABELS = (ADS_LEVEL, ADAS_LEVEL)

LEAKAGE_COLUMNS = frozenset(
    {
        REPORTED_AUTOMATION_LEVEL,
        SOURCE_FILE,
        AUTOMATION_SYSTEM_ENGAGED,
        REPORT_ID,
        # TODO: consider non-leaking narrative-derived features for Phase 1
        # (deliberately deferred; narrative text is excluded for now).
        NARRATIVE,
        ADS_EQUIPPED,
    }
)

BASE_NUMERIC_FEATURES = (
    SV_PRECRASH_SPEED,
    "Posted Speed Limit (MPH)",
)

BASE_CATEGORICAL_FEATURES = (
    DRIVER_OPERATOR_TYPE,
    ROADWAY_TYPE,
    "Roadway Surface",
    "Roadway Description",
    "Lighting",
    "Weather - Clear",
    "Weather - Snow",
    "Weather - Cloudy",
    "Weather - Fog/Smoke",
    "Weather - Rain",
    "Weather - Severe Wind",
    "Weather - Unknown",
    "Weather - Other",
    CRASH_WITH,
    HIGHEST_INJURY_SEVERITY,
    "Property Damage?",
    "CP Pre-Crash Movement",
    SV_PRE_CRASH_MOVEMENT,
    "SV Any Air Bags Deployed?",
    "SV Was Vehicle Towed?",
    "SV Were All Passengers Belted?",
    WITHIN_ODD,
)

HEURISTIC_BOOLEAN_FEATURES = tuple(rule.__name__ for rule in RULES) + ("any_heuristic_flag",)
HEURISTIC_OUTPUT_COLUMNS = HEURISTIC_BOOLEAN_FEATURES + (
    "heuristic_flag_count",
    "rules_triggered",
    "reasons",
)


@dataclass(frozen=True)
class FeatureSpec:
    """Structured model inputs selected from the available Phase 0 columns."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    heuristic_boolean: tuple[str, ...]
    excluded_leakage: tuple[str, ...]
    excluded_high_risk: tuple[str, ...]

    @property
    def selected_columns(self) -> tuple[str, ...]:
        return self.numeric + self.categorical + self.heuristic_boolean


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalars into JSON-serializable values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if pd.isna(value):
        return None
    return value


def _present(columns: Iterable[str], frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(column for column in columns if column in frame.columns)


def _present_numeric_with_values(columns: Iterable[str], frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in columns
        if column in frame.columns and pd.to_numeric(frame[column], errors="coerce").notna().any()
    )


def _present_categorical_with_values(columns: Iterable[str], frame: pd.DataFrame) -> tuple[str, ...]:
    present: list[str] = []
    for column in columns:
        if column not in frame.columns:
            continue
        normalized = frame[column].astype("string").str.strip().replace("", pd.NA)
        if normalized.notna().any():
            present.append(column)
    return tuple(present)


def build_feature_spec(frame: pd.DataFrame) -> FeatureSpec:
    """Select the non-leaking Phase 1 feature columns present in ``frame``.

    ``ADS Equipped?`` is intentionally excluded. In the public files it is
    highly correlated with ADS-source membership and is too close to the label
    for a conservative classical model.
    """

    numeric = _present_numeric_with_values(BASE_NUMERIC_FEATURES, frame)
    categorical = _present_categorical_with_values(BASE_CATEGORICAL_FEATURES, frame)
    heuristic_boolean = _present(HEURISTIC_BOOLEAN_FEATURES, frame)
    selected = set(numeric + categorical + heuristic_boolean)
    leaked = tuple(sorted(selected & LEAKAGE_COLUMNS))
    if leaked:
        raise ValueError(f"Feature selection includes leakage columns: {leaked}")
    if not selected:
        raise ValueError("No Phase 1 feature columns are available in the input frame.")
    return FeatureSpec(
        numeric=numeric,
        categorical=categorical,
        heuristic_boolean=heuristic_boolean,
        excluded_leakage=tuple(sorted(column for column in LEAKAGE_COLUMNS if column in frame.columns)),
        excluded_high_risk=(ADS_EQUIPPED,) if ADS_EQUIPPED in frame.columns else (),
    )


def validate_labels(frame: pd.DataFrame) -> pd.Series:
    """Return the reported labels after checking Phase 1's binary scope."""

    if REPORTED_AUTOMATION_LEVEL not in frame.columns:
        raise ValueError(f"Input frame must include {REPORTED_AUTOMATION_LEVEL!r}.")
    labels = frame[REPORTED_AUTOMATION_LEVEL].astype("string")
    observed = set(labels.dropna().unique())
    expected = set(LABELS)
    unexpected = observed - expected
    if unexpected:
        raise ValueError(f"Unexpected automation labels for Phase 1 classifier: {sorted(unexpected)}")
    if len(observed) < 2:
        raise ValueError("Phase 1 classifier requires both ADS and Level 2 ADAS labels.")
    return labels


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    """Convert heuristic boolean-like values to numeric 0/1 features."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    normalized = series.astype("string").str.strip().str.casefold()
    truthy = {"1", "true", "t", "yes", "y"}
    falsy = {"0", "false", "f", "no", "n", ""}
    mapped = normalized.map(lambda value: 1.0 if value in truthy else 0.0 if value in falsy else np.nan)
    return mapped.fillna(0).astype(float)


def build_feature_matrix(frame: pd.DataFrame, spec: FeatureSpec | None = None) -> tuple[pd.DataFrame, FeatureSpec]:
    """Return model-ready feature columns and the feature specification used."""

    feature_spec = spec or build_feature_spec(frame)
    missing = [column for column in feature_spec.selected_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Input frame is missing selected feature columns: {missing}")

    matrix = pd.DataFrame(index=frame.index)
    for column in feature_spec.numeric:
        matrix[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in feature_spec.categorical:
        categorical = frame[column].astype("string").str.strip().replace("", np.nan).astype(object)
        matrix[column] = categorical.where(~pd.isna(categorical), np.nan)
    for column in feature_spec.heuristic_boolean:
        matrix[column] = _coerce_bool_series(frame[column])
    return matrix.loc[:, feature_spec.selected_columns], feature_spec


def make_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    """Build a preprocessing transformer for the selected feature types."""

    numeric_features = spec.numeric + spec.heuristic_boolean
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_features),
            )
        )
    if spec.categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="__missing__",
                                keep_empty_features=True,
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(spec.categorical),
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def make_logistic_pipeline(spec: FeatureSpec, random_state: int = DEFAULT_RANDOM_STATE) -> Pipeline:
    """Build the class-weighted, interpretable logistic regression baseline."""

    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(spec)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _gradient_boosting_model(random_state: int = DEFAULT_RANDOM_STATE) -> tuple[str, Any]:
    """Return the preferred installed gradient-boosting estimator."""

    try:
        from lightgbm import LGBMClassifier

        return (
            "lightgbm",
            LGBMClassifier(
                class_weight="balanced",
                learning_rate=0.05,
                n_estimators=180,
                num_leaves=15,
                random_state=random_state,
                n_jobs=1,
                verbosity=-1,
            ),
        )
    except Exception:  # pragma: no cover - environment-dependent fallback
        from sklearn.ensemble import HistGradientBoostingClassifier

        return (
            "hist_gradient_boosting_fallback",
            HistGradientBoostingClassifier(
                class_weight="balanced",
                learning_rate=0.05,
                max_iter=180,
                random_state=random_state,
            ),
        )


def make_gradient_boosting_pipeline(spec: FeatureSpec, random_state: int = DEFAULT_RANDOM_STATE) -> tuple[str, Pipeline]:
    """Build the stronger non-linear model, preferring LightGBM when installed."""

    model_name, estimator = _gradient_boosting_model(random_state=random_state)
    return (
        model_name,
        Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(spec)),
                ("model", estimator),
            ]
        ),
    )


def calibrate_model_step(
    pipeline: Pipeline,
    *,
    method: str,
    cv: int = CALIBRATION_CV,
) -> Pipeline:
    """Wrap a pipeline's final estimator in probability calibration.

    Only the estimator step is wrapped, so the returned object stays a
    ``Pipeline`` with a ``model`` step that still exposes ``classes_`` and
    ``predict_proba``. Preprocessing runs before calibration's internal
    cross-validation, which fits the base model and the calibrator on the
    already-transformed features. ``method='none'`` returns the pipeline
    unchanged so default behavior is preserved.
    """

    if method == "none":
        return pipeline
    if method not in CALIBRATION_CHOICES:
        raise ValueError(f"Unsupported calibration method {method!r}; choose one of {CALIBRATION_CHOICES}.")
    calibrated = clone(pipeline)
    base_step_name, base_estimator = calibrated.steps[-1]
    calibrated.steps[-1] = (
        base_step_name,
        CalibratedClassifierCV(base_estimator, method=method, cv=cv),
    )
    return calibrated


def predict_with_confidence(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """Return predicted labels and max-class confidence for ``features``."""

    probabilities = model.predict_proba(features)
    classes = np.asarray(model.named_steps["model"].classes_)
    prediction_indices = probabilities.argmax(axis=1)
    predicted_labels = classes[prediction_indices]
    confidence = probabilities[np.arange(len(probabilities)), prediction_indices]
    return pd.DataFrame(
        {
            "predicted_automation_level": predicted_labels,
            "prediction_confidence": confidence,
        },
        index=features.index,
    )


def evaluate_model(model: Pipeline, features: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
    """Evaluate a fitted model against the reported-label holdout set."""

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    classes = list(model.named_steps["model"].classes_)
    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "class_labels": classes,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(LABELS)).tolist(),
        "classification_report": classification_report(labels, predictions, output_dict=True, zero_division=0),
    }
    try:
        metrics["log_loss"] = log_loss(labels, probabilities, labels=classes)
    except ValueError:
        metrics["log_loss"] = None
    return _json_safe(metrics)


def sweep_flag_thresholds(
    reported_labels: Iterable[str],
    predicted_labels: Iterable[str],
    confidences: Iterable[float],
    *,
    truth_is_misclassified: Iterable[bool] | None = None,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Compute review-flag precision/recall over confidence thresholds.

    When ``truth_is_misclassified`` is provided, precision/recall are measured
    against known positives such as synthetic label flips. Without it, the
    sweep still reports the number and rate of model/reported-label
    disagreements at each threshold.
    """

    reported = np.asarray(list(reported_labels), dtype=object)
    predicted = np.asarray(list(predicted_labels), dtype=object)
    confidence = np.asarray(list(confidences), dtype=float)
    if thresholds is None:
        thresholds = np.round(np.linspace(0.50, 0.99, 50), 2)
    threshold_values = [float(threshold) for threshold in thresholds]
    truth = None if truth_is_misclassified is None else np.asarray(list(truth_is_misclassified), dtype=bool)
    positives = int(truth.sum()) if truth is not None else None

    rows: list[dict[str, Any]] = []
    disagreement = predicted != reported
    for threshold in threshold_values:
        flagged = disagreement & (confidence >= threshold)
        row: dict[str, Any] = {
            "threshold": threshold,
            "flagged_count": int(flagged.sum()),
            "flagged_rate": float(flagged.mean()) if len(flagged) else 0.0,
            "mean_flag_confidence": float(confidence[flagged].mean()) if flagged.any() else None,
        }
        if truth is not None:
            true_positives = flagged & truth
            false_positives = flagged & ~truth
            false_negatives = ~flagged & truth
            precision = float(true_positives.sum() / flagged.sum()) if flagged.any() else 0.0
            recall = float(true_positives.sum() / positives) if positives else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
            row.update(
                {
                    "known_positive_count": positives,
                    "true_positive_count": int(true_positives.sum()),
                    "false_positive_count": int(false_positives.sum()),
                    "false_negative_count": int(false_negatives.sum()),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def choose_operating_point(
    sweep: pd.DataFrame,
    *,
    target_precision: float = DEFAULT_TARGET_SYNTHETIC_PRECISION,
) -> dict[str, Any]:
    """Choose a review threshold from the synthetic mislabel PR sweep."""

    candidates = sweep[(sweep["flagged_count"] > 0) & (sweep["precision"] >= target_precision)]
    if not candidates.empty:
        ordered = candidates.sort_values(
            ["recall", "precision", "threshold"],
            ascending=[False, False, True],
            kind="stable",
        )
        selected = ordered.iloc[0]
        rationale = (
            "Selected the threshold that meets "
            f"the synthetic mislabel precision target of {target_precision:.0%}, "
            "maximizing recall within that precision constraint."
        )
    else:
        nonempty = sweep[sweep["flagged_count"] > 0]
        if nonempty.empty:
            selected = sweep.iloc[-1]
            rationale = "No threshold produced review flags in calibration; selected the highest threshold."
        else:
            selected = nonempty.sort_values(
                ["f1", "precision", "threshold"],
                ascending=[False, False, True],
                kind="stable",
            ).iloc[0]
            rationale = (
                "No calibration point met the target precision, so selected the "
                "non-empty threshold with the best synthetic F1 score."
            )
    result = selected.to_dict()
    result["rationale"] = rationale
    result["target_precision"] = target_precision
    return _json_safe(result)


def _flip_labels_for_threshold_calibration(
    frame: pd.DataFrame,
    *,
    flip_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create deterministic synthetic label flips for calibration folds."""

    rng = np.random.default_rng(random_state)
    perturbed = frame.copy()
    is_flipped = pd.Series(False, index=perturbed.index)

    for label in LABELS:
        label_index = perturbed.index[perturbed[REPORTED_AUTOMATION_LEVEL] == label].to_numpy()
        if len(label_index) == 0:
            continue
        flip_count = int(round(len(label_index) * flip_fraction))
        if flip_fraction > 0 and flip_count == 0:
            flip_count = 1
        flip_count = min(flip_count, len(label_index))
        if flip_count == 0:
            continue
        selected = rng.choice(label_index, size=flip_count, replace=False)
        new_label = ADAS_LEVEL if label == ADS_LEVEL else ADS_LEVEL
        perturbed.loc[selected, REPORTED_AUTOMATION_LEVEL] = new_label
        is_flipped.loc[selected] = True

    return perturbed, is_flipped


def recompute_heuristic_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Re-score Phase 0 heuristic columns after synthetic label changes."""

    without_existing = frame.drop(columns=[column for column in HEURISTIC_OUTPUT_COLUMNS if column in frame.columns])
    return score_records(without_existing)


def calibrate_threshold_with_synthetic_flips(
    frame: pd.DataFrame,
    *,
    model: Pipeline,
    spec: FeatureSpec,
    thresholds: Iterable[float] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_splits: int = 5,
    flip_fraction: float = DEFAULT_SYNTHETIC_FLIP_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use stratified CV with injected label flips to estimate flag PR tradeoffs."""

    labels = validate_labels(frame)
    class_counts = labels.value_counts()
    effective_splits = min(n_splits, int(class_counts.min()))
    if effective_splits < 2:
        raise ValueError("At least two records per class are required for stratified threshold calibration.")

    predictions: list[pd.DataFrame] = []
    cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
    for fold_idx, (train_idx, validation_idx) in enumerate(cv.split(frame, labels), start=1):
        train_frame = frame.iloc[train_idx].copy()
        validation_frame = frame.iloc[validation_idx].copy()

        train_features, _ = build_feature_matrix(train_frame, spec)
        train_labels = train_frame[REPORTED_AUTOMATION_LEVEL].astype("string")
        fold_model = clone(model)
        fold_model.fit(train_features, train_labels)

        perturbed_validation, truth_flipped = _flip_labels_for_threshold_calibration(
            validation_frame,
            flip_fraction=flip_fraction,
            random_state=random_state + fold_idx,
        )
        perturbed_validation = recompute_heuristic_features(perturbed_validation)
        validation_features, _ = build_feature_matrix(perturbed_validation, spec)
        fold_predictions = predict_with_confidence(fold_model, validation_features)
        fold_predictions[REPORTED_AUTOMATION_LEVEL] = perturbed_validation[REPORTED_AUTOMATION_LEVEL].to_numpy()
        fold_predictions["synthetic_mislabel"] = truth_flipped.to_numpy()
        fold_predictions["fold"] = fold_idx
        predictions.append(fold_predictions)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    sweep = sweep_flag_thresholds(
        prediction_frame[REPORTED_AUTOMATION_LEVEL],
        prediction_frame["predicted_automation_level"],
        prediction_frame["prediction_confidence"],
        truth_is_misclassified=prediction_frame["synthetic_mislabel"],
        thresholds=thresholds,
    )
    return sweep, prediction_frame


def build_flagged_records(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    threshold: float,
    model_name: str,
) -> pd.DataFrame:
    """Return ranked model/reported-label disagreements for human review."""

    result = frame.copy()
    result["classifier_model"] = model_name
    result["predicted_automation_level"] = predictions["predicted_automation_level"].to_numpy()
    result["disagreement_confidence"] = predictions["prediction_confidence"].to_numpy()
    result["classifier_disagrees_with_reported_label"] = (
        result["predicted_automation_level"] != result[REPORTED_AUTOMATION_LEVEL]
    )
    result["classifier_flag_threshold"] = threshold
    result["classifier_flag_for_review"] = (
        result["classifier_disagrees_with_reported_label"] & (result["disagreement_confidence"] >= threshold)
    )

    if "rules_triggered" in result.columns:
        result["top_contributing_heuristic_flags"] = result["rules_triggered"].fillna("")
    else:
        summaries = []
        for _, row in result.iterrows():
            summaries.append(
                "; ".join(
                    column for column in HEURISTIC_BOOLEAN_FEATURES if column in result.columns and bool(row[column])
                )
            )
        result["top_contributing_heuristic_flags"] = summaries
    if "reasons" in result.columns:
        result["heuristic_reasons"] = result["reasons"].fillna("")
    else:
        result["heuristic_reasons"] = ""

    output_columns = [
        REPORT_ID,
        REPORTING_ENTITY,
        REPORTED_AUTOMATION_LEVEL,
        "predicted_automation_level",
        "disagreement_confidence",
        "classifier_model",
        "classifier_flag_threshold",
        "top_contributing_heuristic_flags",
        "heuristic_reasons",
    ]
    available_columns = [column for column in output_columns if column in result.columns]
    flagged = result.loc[result["classifier_flag_for_review"], available_columns].copy()
    return flagged.sort_values("disagreement_confidence", ascending=False, kind="stable").reset_index(drop=True)


def build_prediction_records(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    threshold: float,
    model_name: str,
) -> pd.DataFrame:
    """Return compact Phase 1 per-row predictions for Phase 2 comparison."""

    result = pd.DataFrame(index=frame.index)
    for column in [REPORT_ID, REPORTING_ENTITY, REPORTED_AUTOMATION_LEVEL]:
        if column in frame.columns:
            result[column] = frame[column].to_numpy()
    result["predicted_automation_level"] = predictions["predicted_automation_level"].to_numpy()
    result["prediction_confidence"] = predictions["prediction_confidence"].to_numpy()
    result["classifier_model"] = model_name
    result["classifier_flag_threshold"] = threshold
    if REPORTED_AUTOMATION_LEVEL in result.columns:
        result["classifier_disagrees_with_reported_label"] = (
            result["predicted_automation_level"] != result[REPORTED_AUTOMATION_LEVEL]
        )
        result["classifier_flag_for_review"] = (
            result["classifier_disagrees_with_reported_label"] & (result["prediction_confidence"] >= threshold)
        )
    result["responsible_use"] = "Rows are review signals only, not confirmed misclassifications or safety-rate claims."
    return result.reset_index(drop=True)


def out_of_fold_predictions(
    frame: pd.DataFrame,
    *,
    model: Pipeline,
    spec: FeatureSpec,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Generate out-of-fold predictions for ranking all current records."""

    labels = validate_labels(frame)
    class_counts = labels.value_counts()
    effective_splits = min(n_splits, int(class_counts.min()))
    if effective_splits < 2:
        raise ValueError("At least two records per class are required for out-of-fold predictions.")

    predictions = pd.DataFrame(index=frame.index)
    cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
    for train_idx, validation_idx in cv.split(frame, labels):
        train_frame = frame.iloc[train_idx].copy()
        validation_frame = frame.iloc[validation_idx].copy()
        train_features, _ = build_feature_matrix(train_frame, spec)
        train_labels = train_frame[REPORTED_AUTOMATION_LEVEL].astype("string")
        validation_features, _ = build_feature_matrix(validation_frame, spec)
        fold_model = clone(model)
        fold_model.fit(train_features, train_labels)
        predictions.loc[validation_frame.index, ["predicted_automation_level", "prediction_confidence"]] = (
            predict_with_confidence(fold_model, validation_features)
        )
    return predictions.loc[frame.index]


def logistic_feature_importance(model: Pipeline, *, top_n: int = 25) -> pd.DataFrame:
    """Return absolute logistic-regression coefficient magnitudes."""

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = estimator.coef_[0]
    positive_class = estimator.classes_[1] if len(estimator.classes_) > 1 else estimator.classes_[0]
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
            "positive_class": positive_class,
        }
    )
    return (
        importance.sort_values("absolute_coefficient", ascending=False, kind="stable")
        .head(top_n)
        .reset_index(drop=True)
    )


def plot_precision_recall_curve(sweep: pd.DataFrame, operating_point: dict[str, Any], output_path: Path) -> None:
    """Write the synthetic mislabel precision/recall chart."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(sweep["recall"], sweep["precision"], marker="o", linewidth=1.5)
    plt.scatter(
        [operating_point["recall"]],
        [operating_point["precision"]],
        color="red",
        label=f"chosen threshold={operating_point['threshold']:.2f}",
        zorder=3,
    )
    plt.xlabel("Recall on synthetic label flips")
    plt.ylabel("Precision on synthetic label flips")
    plt.title("Classifier review-flag threshold sweep")
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_feature_importance(importance: pd.DataFrame, output_path: Path) -> None:
    """Write the logistic-regression coefficient chart."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = importance.sort_values("absolute_coefficient", ascending=True)
    plt.figure(figsize=(8, max(5, 0.28 * len(ordered))))
    plt.barh(ordered["feature"], ordered["absolute_coefficient"])
    plt.xlabel("Absolute logistic-regression coefficient")
    plt.title("Top structured-feature contributions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def load_phase0_frame(input_path: Path, combined_fallback_path: Path = DEFAULT_COMBINED_INPUT_PATH) -> pd.DataFrame:
    """Load Phase 0 scored records, scoring combined records if needed."""

    if input_path.exists():
        return pd.read_csv(input_path, dtype="string", keep_default_na=False, na_values=[])
    if combined_fallback_path.exists():
        combined = pd.read_csv(combined_fallback_path, dtype="string", keep_default_na=False, na_values=[])
        return score_records(combined)
    raise FileNotFoundError(
        f"Neither {input_path} nor fallback {combined_fallback_path} exists. Run Phase 0 first."
    )


def train_evaluate_and_write_artifacts(
    frame: pd.DataFrame,
    *,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_size: float = DEFAULT_TEST_SIZE,
    target_precision: float = DEFAULT_TARGET_SYNTHETIC_PRECISION,
    threshold_override: float | None = None,
    calibration: str = DEFAULT_CALIBRATION,
) -> dict[str, Any]:
    """Train both Phase 1 models, evaluate, and write requested artifacts.

    ``calibration`` optionally wraps each model's probabilities in
    ``CalibratedClassifierCV`` ('sigmoid' or 'isotonic') so the disagreement
    confidence used for review-flag thresholds is better calibrated. The
    default 'none' preserves the historical, uncalibrated behavior.
    """

    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    labels = validate_labels(frame)
    spec = build_feature_spec(frame)
    features, _ = build_feature_matrix(frame, spec)

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    logistic_model = make_logistic_pipeline(spec, random_state=random_state)
    gradient_name, gradient_model = make_gradient_boosting_pipeline(spec, random_state=random_state)
    model_specs = {
        "logistic_regression": calibrate_model_step(logistic_model, method=calibration),
        gradient_name: calibrate_model_step(gradient_model, method=calibration),
    }

    fitted_models: dict[str, Pipeline] = {}
    model_metrics: dict[str, Any] = {}
    for model_name, model in model_specs.items():
        fitted = clone(model)
        fitted.fit(train_features, train_labels)
        fitted_models[model_name] = fitted
        model_metrics[model_name] = evaluate_model(fitted, test_features, test_labels)

    selected_model_name = max(
        model_metrics,
        key=lambda name: (
            model_metrics[name]["f1_macro"],
            model_metrics[name]["balanced_accuracy"],
        ),
    )
    selected_model_template = model_specs[selected_model_name]

    threshold_sweep, calibration_predictions = calibrate_threshold_with_synthetic_flips(
        frame,
        model=selected_model_template,
        spec=spec,
        random_state=random_state,
        flip_fraction=DEFAULT_SYNTHETIC_FLIP_FRACTION,
    )
    operating_point = choose_operating_point(threshold_sweep, target_precision=target_precision)
    if threshold_override is not None:
        matching = threshold_sweep.iloc[(threshold_sweep["threshold"] - threshold_override).abs().argsort()].iloc[0]
        operating_point = matching.to_dict()
        operating_point["rationale"] = "Threshold was supplied explicitly on the command line."
        operating_point["target_precision"] = target_precision
    threshold = float(operating_point["threshold"])

    selected_oof_predictions = out_of_fold_predictions(
        frame,
        model=selected_model_template,
        spec=spec,
        random_state=random_state,
    )
    flagged = build_flagged_records(
        frame,
        selected_oof_predictions,
        threshold=threshold,
        model_name=selected_model_name,
    )
    prediction_records = build_prediction_records(
        frame,
        selected_oof_predictions,
        threshold=threshold,
        model_name=selected_model_name,
    )

    importance_model = fitted_models["logistic_regression"]
    if not hasattr(importance_model.named_steps["model"], "coef_"):
        importance_model = clone(logistic_model)
        importance_model.fit(train_features, train_labels)
    logistic_importance = logistic_feature_importance(importance_model)

    flagged_path = processed_dir / "classifier_flagged_records.csv"
    predictions_path = processed_dir / "classifier_predictions.csv"
    metrics_path = processed_dir / "classifier_metrics.json"
    operating_point_path = processed_dir / "classifier_operating_point.json"
    threshold_sweep_path = processed_dir / "classifier_threshold_sweep.csv"
    calibration_predictions_path = processed_dir / "classifier_threshold_calibration_predictions.csv"
    feature_importance_csv_path = processed_dir / "classifier_logistic_feature_importance.csv"
    pr_curve_path = outputs_dir / "classifier_precision_recall_curve.png"
    feature_importance_chart_path = outputs_dir / "classifier_logistic_feature_importance.png"
    operating_point_note_path = outputs_dir / "classifier_operating_point.md"

    flagged.to_csv(flagged_path, index=False)
    prediction_records.to_csv(predictions_path, index=False)
    threshold_sweep.to_csv(threshold_sweep_path, index=False)
    calibration_predictions.to_csv(calibration_predictions_path, index=False)
    logistic_importance.to_csv(feature_importance_csv_path, index=False)
    plot_precision_recall_curve(threshold_sweep, operating_point, pr_curve_path)
    plot_feature_importance(logistic_importance, feature_importance_chart_path)

    operating_point_note = (
        "# Classifier operating point\n\n"
        f"- Recommended threshold: `{threshold:.2f}`\n"
        f"- Calibration method: stratified cross-validation with "
        f"{DEFAULT_SYNTHETIC_FLIP_FRACTION:.0%} synthetic label flips per class.\n"
        f"- Precision on synthetic flips: `{operating_point.get('precision', 0):.3f}`\n"
        f"- Recall on synthetic flips: `{operating_point.get('recall', 0):.3f}`\n"
        f"- Rationale: {operating_point['rationale']}\n\n"
        "These calibration labels are synthetic perturbations, not ground truth "
        "for real NHTSA records. Real output rows are only flagged for human review.\n"
    )
    operating_point_note_path.write_text(operating_point_note, encoding="utf-8")

    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(frame)),
        "label_distribution": labels.value_counts().to_dict(),
        "random_state": random_state,
        "test_size": test_size,
        "calibration": calibration,
        "probability_calibration": {
            "method": calibration,
            "cv_folds": CALIBRATION_CV if calibration != "none" else None,
            "note": (
                "Model probabilities are wrapped in CalibratedClassifierCV so the "
                "disagreement-confidence threshold reflects calibrated probabilities."
                if calibration != "none"
                else "Uncalibrated model probabilities (historical default)."
            ),
        },
        "class_imbalance_handling": {
            "split": "stratified train/test split",
            "threshold_selection": "stratified cross-validation with synthetic label flips",
            "model_weights": "class_weight='balanced' for logistic regression and LightGBM/fallback gradient boosting",
        },
        "feature_policy": {
            "selected_numeric": spec.numeric,
            "selected_categorical": spec.categorical,
            "selected_heuristic_boolean": spec.heuristic_boolean,
            "excluded_leakage": spec.excluded_leakage,
            "ads_equipped_policy": "excluded as high-risk near-leak/source-membership proxy",
            "excluded_high_risk": spec.excluded_high_risk,
        },
        "models": model_metrics,
        "selected_flag_model": selected_model_name,
        "operating_point": operating_point,
        "flagged_records": {
            "count": int(len(flagged)),
            "path": str(flagged_path),
            "review_language": "Rows are flagged for review only; they are not findings of misclassification.",
        },
        "prediction_records": {
            "count": int(len(prediction_records)),
            "path": str(predictions_path),
            "review_language": "Rows are review signals only; they are not confirmed misclassifications.",
        },
        "artifacts": {
            "metrics_json": str(metrics_path),
            "predictions_csv": str(predictions_path),
            "operating_point_json": str(operating_point_path),
            "operating_point_note": str(operating_point_note_path),
            "threshold_sweep_csv": str(threshold_sweep_path),
            "threshold_calibration_predictions_csv": str(calibration_predictions_path),
            "precision_recall_curve_png": str(pr_curve_path),
            "logistic_feature_importance_csv": str(feature_importance_csv_path),
            "logistic_feature_importance_png": str(feature_importance_chart_path),
        },
    }

    metrics_path.write_text(json.dumps(_json_safe(metrics), indent=2, sort_keys=True), encoding="utf-8")
    operating_point_payload = {
        "selected_flag_model": selected_model_name,
        "operating_point": operating_point,
        "methodology": metrics["class_imbalance_handling"],
        "responsible_use": "Flags are review candidates only, not confirmed misclassifications or safety-rate claims.",
    }
    operating_point_path.write_text(
        json.dumps(_json_safe(operating_point_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return _json_safe(metrics)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for ``python -m src.classifier``."""

    parser = argparse.ArgumentParser(description="Train/evaluate Phase 1 SGO classical classifiers.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--combined-fallback", type=Path, default=DEFAULT_COMBINED_INPUT_PATH)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--target-precision", type=float, default=DEFAULT_TARGET_SYNTHETIC_PRECISION)
    parser.add_argument("--threshold", type=float, default=None, help="Optional explicit disagreement-confidence threshold.")
    parser.add_argument(
        "--calibration",
        choices=CALIBRATION_CHOICES,
        default=DEFAULT_CALIBRATION,
        help=(
            "Probability calibration for the disagreement-confidence score. "
            "'none' (default) keeps historical behavior; 'sigmoid' or 'isotonic' "
            "wrap the models in CalibratedClassifierCV."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Phase 1 classifier pipeline from the command line."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    frame = load_phase0_frame(args.input, args.combined_fallback)
    metrics = train_evaluate_and_write_artifacts(
        frame,
        processed_dir=args.processed_dir,
        outputs_dir=args.outputs_dir,
        random_state=args.random_state,
        test_size=args.test_size,
        target_precision=args.target_precision,
        threshold_override=args.threshold,
        calibration=args.calibration,
    )
    summary = {
        "rows": metrics["row_count"],
        "selected_flag_model": metrics["selected_flag_model"],
        "calibration": metrics["calibration"],
        "model_scores": {
            name: {
                "balanced_accuracy": values["balanced_accuracy"],
                "f1_macro": values["f1_macro"],
            }
            for name, values in metrics["models"].items()
        },
        "operating_point": metrics["operating_point"],
        "flagged_records": metrics["flagged_records"],
    }
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
