"""Phase 2 three-way disagreement analysis.

Compares the reported SGO automation label, the Phase 1 structured classifier
prediction, and the Phase 2 narrative inference. The resulting categories are
review priorities only; they are not confirmed misclassifications and make no
manufacturer safety ranking or safety-rate claim.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .fields import REPORT_ID, REPORTED_AUTOMATION_LEVEL, REPORTING_ENTITY
from .llm_auditor import RESPONSIBLE_USE_NOTE

DEFAULT_PHASE1_PREDICTIONS_PATH = Path("data/processed/classifier_predictions.csv")
DEFAULT_PHASE1_FLAGGED_PATH = Path("data/processed/classifier_flagged_records.csv")
DEFAULT_PHASE2_PATH = Path("data/processed/llm_audit_results.csv")
DEFAULT_COMPARISON_PATH = Path("data/processed/phase_comparison.csv")
DEFAULT_MATRIX_PATH = Path("data/processed/disagreement_matrix.json")
DEFAULT_CHART_PATH = Path("outputs/disagreement_breakdown.png")

CATEGORY_DEFINITIONS = {
    "all_agree": {
        "priority": "low",
        "definition": "Reported label, Phase 1 prediction, and Phase 2 narrative inference all match.",
    },
    "phase1_phase2_agree_vs_reported": {
        "priority": "highest",
        "definition": (
            "Phase 1 and Phase 2 independently agree with each other and differ "
            "from the reported label; flagged for human review."
        ),
    },
    "phase1_phase2_disagree": {
        "priority": "ambiguous",
        "definition": (
            "Phase 1 and Phase 2 disagree with each other; interesting or ambiguous, "
            "but not a strong standalone flag."
        ),
    },
    "only_one_method_available": {
        "priority": "limited",
        "definition": "One or more required labels are unavailable for a three-way comparison.",
    },
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype="string", keep_default_na=False, na_values=[])


def _blank_to_na(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace("", pd.NA)


def load_phase1_predictions(predictions_path: Path, flagged_path: Path) -> pd.DataFrame:
    """Load preferred full Phase 1 predictions or fall back to flagged subset."""

    if predictions_path.exists():
        frame = _read_csv(predictions_path)
        source = str(predictions_path)
    elif flagged_path.exists():
        frame = _read_csv(flagged_path)
        source = str(flagged_path)
    else:
        raise FileNotFoundError(
            f"Neither Phase 1 predictions {predictions_path} nor flagged subset {flagged_path} exists."
        )

    if REPORT_ID not in frame.columns:
        raise ValueError(f"Phase 1 input {source} must include {REPORT_ID!r}.")
    prediction_column = (
        "phase1_predicted_automation_level"
        if "phase1_predicted_automation_level" in frame.columns
        else "predicted_automation_level"
    )
    confidence_column = (
        "phase1_prediction_confidence"
        if "phase1_prediction_confidence" in frame.columns
        else "prediction_confidence"
        if "prediction_confidence" in frame.columns
        else "disagreement_confidence"
        if "disagreement_confidence" in frame.columns
        else None
    )
    if prediction_column not in frame.columns:
        raise ValueError(f"Phase 1 input {source} must include a predicted automation-level column.")

    output = pd.DataFrame(
        {
            REPORT_ID: frame[REPORT_ID],
            "phase1_predicted_automation_level": frame[prediction_column],
            "phase1_source": source,
        }
    )
    if confidence_column is not None:
        output["phase1_prediction_confidence"] = frame[confidence_column]
    else:
        output["phase1_prediction_confidence"] = pd.NA
    for column in [
        REPORTED_AUTOMATION_LEVEL,
        REPORTING_ENTITY,
        "classifier_model",
        "classifier_flag_for_review",
    ]:
        if column in frame.columns:
            output[column] = frame[column]
    return output.drop_duplicates(subset=[REPORT_ID], keep="first")


def load_phase2_results(path: Path) -> pd.DataFrame:
    """Load Phase 2 narrative inference results."""

    frame = _read_csv(path)
    required = {REPORT_ID, "llm_inferred_automation_level", "inference_backend"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Phase 2 input {path} is missing required columns: {sorted(missing)}")
    output = pd.DataFrame(
        {
            REPORT_ID: frame[REPORT_ID],
            "phase2_llm_inferred_automation_level": frame["llm_inferred_automation_level"],
            "phase2_llm_rationale": frame["llm_rationale"] if "llm_rationale" in frame.columns else pd.NA,
            "phase2_inference_backend": frame["inference_backend"],
            "phase2_inference_status": frame["inference_status"] if "inference_status" in frame.columns else pd.NA,
            "phase2_model_name": frame["model_name"] if "model_name" in frame.columns else pd.NA,
        }
    )
    for column in [REPORTED_AUTOMATION_LEVEL, REPORTING_ENTITY]:
        if column in frame.columns:
            output[column] = frame[column]
    return output.drop_duplicates(subset=[REPORT_ID], keep="first")


def categorize_row(row: pd.Series) -> str:
    """Assign the requested Phase 2 disagreement-matrix category."""

    reported = row.get(REPORTED_AUTOMATION_LEVEL)
    phase1 = row.get("phase1_predicted_automation_level")
    phase2 = row.get("phase2_llm_inferred_automation_level")
    if pd.isna(reported) or pd.isna(phase1) or pd.isna(phase2):
        return "only_one_method_available"
    reported_text = str(reported).strip()
    phase1_text = str(phase1).strip()
    phase2_text = str(phase2).strip()
    if not reported_text or not phase1_text or not phase2_text:
        return "only_one_method_available"
    if reported_text == phase1_text == phase2_text:
        return "all_agree"
    if phase1_text == phase2_text and phase1_text != reported_text:
        return "phase1_phase2_agree_vs_reported"
    return "phase1_phase2_disagree"


def build_comparison(phase1: pd.DataFrame, phase2: pd.DataFrame) -> pd.DataFrame:
    """Join Phase 1 and Phase 2 rows and categorize each available record."""

    comparison = phase2.merge(
        phase1,
        on=REPORT_ID,
        how="left",
        suffixes=("_phase2", "_phase1"),
    )
    if f"{REPORTED_AUTOMATION_LEVEL}_phase2" in comparison.columns:
        comparison[REPORTED_AUTOMATION_LEVEL] = comparison[f"{REPORTED_AUTOMATION_LEVEL}_phase2"]
        phase1_reported = f"{REPORTED_AUTOMATION_LEVEL}_phase1"
        if phase1_reported in comparison.columns:
            comparison[REPORTED_AUTOMATION_LEVEL] = comparison[REPORTED_AUTOMATION_LEVEL].where(
                _blank_to_na(comparison[REPORTED_AUTOMATION_LEVEL]).notna(),
                comparison[phase1_reported],
            )
    elif f"{REPORTED_AUTOMATION_LEVEL}_phase1" in comparison.columns:
        comparison[REPORTED_AUTOMATION_LEVEL] = comparison[f"{REPORTED_AUTOMATION_LEVEL}_phase1"]
    if f"{REPORTING_ENTITY}_phase2" in comparison.columns:
        comparison[REPORTING_ENTITY] = comparison[f"{REPORTING_ENTITY}_phase2"]
        phase1_entity = f"{REPORTING_ENTITY}_phase1"
        if phase1_entity in comparison.columns:
            comparison[REPORTING_ENTITY] = comparison[REPORTING_ENTITY].where(
                _blank_to_na(comparison[REPORTING_ENTITY]).notna(),
                comparison[phase1_entity],
            )
    elif f"{REPORTING_ENTITY}_phase1" in comparison.columns:
        comparison[REPORTING_ENTITY] = comparison[f"{REPORTING_ENTITY}_phase1"]

    for column in [
        REPORTED_AUTOMATION_LEVEL,
        "phase1_predicted_automation_level",
        "phase2_llm_inferred_automation_level",
    ]:
        if column in comparison.columns:
            comparison[column] = _blank_to_na(comparison[column])

    comparison["disagreement_category"] = comparison.apply(categorize_row, axis=1)
    comparison["review_priority"] = comparison["disagreement_category"].map(
        lambda category: CATEGORY_DEFINITIONS[category]["priority"]
    )
    comparison["responsible_use"] = RESPONSIBLE_USE_NOTE

    output_columns = [
        REPORT_ID,
        REPORTING_ENTITY,
        REPORTED_AUTOMATION_LEVEL,
        "phase1_predicted_automation_level",
        "phase1_prediction_confidence",
        "phase2_llm_inferred_automation_level",
        "phase2_inference_backend",
        "phase2_inference_status",
        "phase2_llm_rationale",
        "disagreement_category",
        "review_priority",
        "responsible_use",
    ]
    available_columns = [column for column in output_columns if column in comparison.columns]
    return comparison.loc[:, available_columns].sort_values(
        ["review_priority", "disagreement_category", REPORT_ID],
        kind="stable",
    )


def build_matrix(comparison: pd.DataFrame) -> dict[str, Any]:
    """Build JSON-serializable counts and definitions for the matrix output."""

    counts = {
        category: int((comparison["disagreement_category"] == category).sum())
        for category in CATEGORY_DEFINITIONS
    }
    backend_counts = (
        comparison["phase2_inference_backend"].value_counts(dropna=False).to_dict()
        if "phase2_inference_backend" in comparison.columns
        else {}
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_compared": int(len(comparison)),
        "counts": counts,
        "category_definitions": CATEGORY_DEFINITIONS,
        "phase2_inference_backend_counts": {str(key): int(value) for key, value in backend_counts.items()},
        "responsible_use": RESPONSIBLE_USE_NOTE,
    }


def plot_disagreement_breakdown(comparison: pd.DataFrame, output_path: Path) -> None:
    """Write a small category-count chart for local reporting."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = comparison["disagreement_category"].value_counts().reindex(CATEGORY_DEFINITIONS, fill_value=0)
    plt.figure(figsize=(8, 4.5))
    counts.plot(kind="bar")
    plt.ylabel("Record count")
    plt.title("Phase 2 disagreement matrix (human-review signals)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_comparison(
    *,
    phase1_predictions_path: Path = DEFAULT_PHASE1_PREDICTIONS_PATH,
    phase1_flagged_path: Path = DEFAULT_PHASE1_FLAGGED_PATH,
    phase2_path: Path = DEFAULT_PHASE2_PATH,
    comparison_path: Path = DEFAULT_COMPARISON_PATH,
    matrix_path: Path = DEFAULT_MATRIX_PATH,
    chart_path: Path | None = DEFAULT_CHART_PATH,
) -> dict[str, Any]:
    """Run the Phase 2 comparison and write CSV/JSON/chart artifacts."""

    phase1 = load_phase1_predictions(phase1_predictions_path, phase1_flagged_path)
    phase2 = load_phase2_results(phase2_path)
    comparison = build_comparison(phase1, phase2)
    matrix = build_matrix(comparison)

    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True), encoding="utf-8")
    if chart_path is not None:
        plot_disagreement_breakdown(comparison, chart_path)
        matrix["chart_path"] = str(chart_path)
    matrix["comparison_path"] = str(comparison_path)
    matrix["matrix_path"] = str(matrix_path)
    return matrix


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for ``python -m src.compare``."""

    parser = argparse.ArgumentParser(description="Compare reported, Phase 1, and Phase 2 labels.")
    parser.add_argument("--phase1-predictions", type=Path, default=DEFAULT_PHASE1_PREDICTIONS_PATH)
    parser.add_argument("--phase1-flagged", type=Path, default=DEFAULT_PHASE1_FLAGGED_PATH)
    parser.add_argument("--phase2", type=Path, default=DEFAULT_PHASE2_PATH)
    parser.add_argument("--comparison-output", type=Path, default=DEFAULT_COMPARISON_PATH)
    parser.add_argument("--matrix-output", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--chart-output", type=Path, default=DEFAULT_CHART_PATH)
    parser.add_argument("--no-chart", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_comparison(
        phase1_predictions_path=args.phase1_predictions,
        phase1_flagged_path=args.phase1_flagged,
        phase2_path=args.phase2,
        comparison_path=args.comparison_output,
        matrix_path=args.matrix_output,
        chart_path=None if args.no_chart else args.chart_output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
