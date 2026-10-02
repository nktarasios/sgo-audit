"""Phase 0 reporting-entity classification-consistency trend tracking.

Usage:
    python -m src.entity_trends
    python -m src.entity_trends --input data/processed/heuristic_scored_records.csv

Outputs are data-quality KPIs only. The chart label is intentionally
"classification-consistency flag rate" and must not be interpreted as a safety
incident rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .fields import INCIDENT_DATE, REPORT_MONTH, REPORT_PERIOD, REPORT_YEAR, REPORTING_ENTITY


def _coerce_flag_series(frame: pd.DataFrame, flag_column: str) -> pd.Series:
    if flag_column not in frame.columns:
        raise ValueError(f"Input is missing required flag column: {flag_column}")
    values = frame[flag_column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype("string").str.casefold().isin({"true", "1", "yes", "y"})


def add_trend_period(frame: pd.DataFrame, frequency: str = "M") -> pd.DataFrame:
    """Add a normalized period column from Report Year/Month or Incident Date."""

    frequency = frequency.upper()
    if frequency not in {"M", "Q"}:
        raise ValueError("frequency must be 'M' or 'Q'")

    with_period = frame.copy()
    period = pd.Series(pd.NA, index=with_period.index, dtype="string")

    if REPORT_YEAR in with_period.columns and REPORT_MONTH in with_period.columns:
        year = pd.to_numeric(with_period[REPORT_YEAR], errors="coerce")
        month = pd.to_numeric(with_period[REPORT_MONTH], errors="coerce")
        valid = year.notna() & month.between(1, 12)
        dates = pd.to_datetime(
            {
                "year": year.loc[valid].astype(int),
                "month": month.loc[valid].astype(int),
                "day": 1,
            },
            errors="coerce",
        )
        period.loc[valid] = dates.dt.to_period(frequency).astype(str).to_numpy()

    if period.isna().any() and INCIDENT_DATE in with_period.columns:
        missing = period.isna()
        incident_dates = pd.to_datetime(
            with_period.loc[missing, INCIDENT_DATE],
            errors="coerce",
            format="mixed",
        )
        valid_incident = incident_dates.notna()
        period.loc[incident_dates.loc[valid_incident].index] = (
            incident_dates.loc[valid_incident].dt.to_period(frequency).astype(str).to_numpy()
        )

    with_period[REPORT_PERIOD] = period
    return with_period


def aggregate_entity_flag_rates(
    scored_frame: pd.DataFrame,
    *,
    frequency: str = "M",
    flag_column: str = "any_heuristic_flag",
) -> pd.DataFrame:
    """Aggregate flag counts and rates by Reporting Entity and period."""

    if REPORTING_ENTITY not in scored_frame.columns:
        raise ValueError(f"Input is missing required entity column: {REPORTING_ENTITY}")

    frame = add_trend_period(scored_frame, frequency=frequency)
    frame["_is_flagged"] = _coerce_flag_series(frame, flag_column)
    frame = frame.dropna(subset=[REPORTING_ENTITY, REPORT_PERIOD])

    grouped = (
        frame.groupby([REPORTING_ENTITY, REPORT_PERIOD], dropna=False)
        .agg(total_records=("_is_flagged", "size"), flagged_records=("_is_flagged", "sum"))
        .reset_index()
    )
    grouped["classification_consistency_flag_rate"] = (
        grouped["flagged_records"] / grouped["total_records"]
    )
    return grouped.sort_values([REPORTING_ENTITY, REPORT_PERIOD]).reset_index(drop=True)


def save_entity_trend_chart(
    trends: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n_entities: int = 8,
) -> Path:
    """Save a line chart of classification-consistency flag rate over time."""

    chart_path = Path(output_path)
    chart_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6))
    if trends.empty:
        plt.title("classification-consistency flag rate by reporting entity")
        plt.ylabel("classification-consistency flag rate")
        plt.xlabel("report period")
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150)
        plt.close()
        return chart_path

    entity_order = (
        trends.groupby(REPORTING_ENTITY)["total_records"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n_entities)
        .index
    )
    plot_frame = trends[trends[REPORTING_ENTITY].isin(entity_order)].copy()

    for entity, entity_frame in plot_frame.groupby(REPORTING_ENTITY):
        entity_frame = entity_frame.sort_values(REPORT_PERIOD)
        plt.plot(
            entity_frame[REPORT_PERIOD],
            entity_frame["classification_consistency_flag_rate"],
            marker="o",
            linewidth=1.5,
            label=str(entity),
        )

    plt.title("classification-consistency flag rate by reporting entity")
    plt.ylabel("classification-consistency flag rate")
    plt.xlabel("report period")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(bottom=0)
    plt.legend(title="Reporting Entity", fontsize="small", loc="best")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    return chart_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Phase 0 entity flag-rate trends.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/heuristic_scored_records.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/entity_flag_rates.csv"))
    parser.add_argument("--chart-output", type=Path, default=Path("outputs/entity_flag_rates.png"))
    parser.add_argument("--frequency", choices=["M", "Q"], default="M")
    parser.add_argument("--top-n-entities", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    scored = pd.read_csv(args.input, dtype="string", keep_default_na=False, na_values=[])
    trends = aggregate_entity_flag_rates(scored, frequency=args.frequency)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trends.to_csv(args.output, index=False)
    chart_path = save_entity_trend_chart(
        trends,
        args.chart_output,
        top_n_entities=args.top_n_entities,
    )

    print(
        json.dumps(
            {
                "trend_rows": int(len(trends)),
                "entities": int(trends[REPORTING_ENTITY].nunique()) if not trends.empty else 0,
                "output": str(args.output),
                "chart_output": str(chart_path),
                "metric_label": "classification-consistency flag rate",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
