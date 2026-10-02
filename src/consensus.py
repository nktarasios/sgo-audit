"""Phase 2.5 consensus review queue.

Fuses the three independent SGO-Audit signals into a single ranked review
queue so the highest-value records are the ones where multiple methods each
disagree with the reported automation-level label:

- Phase 0 heuristic rules (``any_heuristic_flag``),
- Phase 1 classifier confident disagreement (``classifier_flag_for_review``),
- Phase 2 narrative LLM inference disagreeing with the reported label.

A record's ``consensus_score`` is the number of these signals (0-3) that
disagree with the reported label. The queue keeps records with at least one
signal and ranks three-signal consensus above two-signal above single-signal.

Consensus rows are review candidates only. Agreement between methods raises
review priority; it is never a confirmed misclassification, a manufacturer
safety ranking, or a safety-rate claim.
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

DEFAULT_PHASE1_PREDICTIONS_PATH = Path("data/processed/classifier_predictions.csv")
DEFAULT_HEURISTIC_SCORED_PATH = Path("data/processed/heuristic_scored_records.csv")
DEFAULT_HEURISTIC_FLAGS_PATH = Path("data/processed/heuristic_flags.csv")
DEFAULT_COMPARISON_PATH = Path("data/processed/phase_comparison.csv")
DEFAULT_QUEUE_PATH = Path("data/processed/consensus_review_queue.csv")
DEFAULT_SUMMARY_PATH = Path("data/processed/consensus_summary.json")
DEFAULT_CHART_PATH = Path("outputs/consensus_tiers.png")

RESPONSIBLE_USE_NOTE = (
    "Consensus rows are review candidates only. Multiple methods disagreeing with "
    "the reported label raises review priority; it is not a confirmed "
    "misclassification, safety ranking, or safety-rate claim."
)

TIER_BY_SCORE = {
    3: "three_signal_consensus",
    2: "two_signal_consensus",
    1: "single_signal",
}

TIER_PRIORITY = {
    "three_signal_consensus": "highest",
    "two_signal_consensus": "elevated",
    "single_signal": "review",
}

TRUTHY = {"1", "true", "t", "yes", "y"}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype="string", keep_default_na=False, na_values=[])


def _to_bool(series: pd.Series) -> pd.Series:
    """Coerce a string/boolean-like column to a clean boolean Series."""

    normalized = series.astype("string").str.strip().str.casefold()
    return normalized.isin(TRUTHY).fillna(False)


def _clean_label(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace("", pd.NA)


def load_heuristic_flags(
    scored_path: Path = DEFAULT_HEURISTIC_SCORED_PATH,
    flags_path: Path = DEFAULT_HEURISTIC_FLAGS_PATH,
) -> pd.DataFrame:
    """Return per-record heuristic disagreement signal keyed by report id.

    Prefers the full scored records; falls back to the flagged-only subset
    (in which case every listed record counts as heuristic-flagged).
    """

    if scored_path.exists():
        frame = _read_csv(scored_path)
        if "any_heuristic_flag" in frame.columns:
            heuristic_disagrees = _to_bool(frame["any_heuristic_flag"])
        else:
            heuristic_disagrees = pd.Series(False, index=frame.index)
        rules = frame["rules_triggered"] if "rules_triggered" in frame.columns else pd.NA
    elif flags_path.exists():
        frame = _read_csv(flags_path)
        heuristic_disagrees = pd.Series(True, index=frame.index)
        rules = frame["rules_triggered"] if "rules_triggered" in frame.columns else pd.NA
    else:
        raise FileNotFoundError(f"Neither {scored_path} nor {flags_path} exists. Run Phase 0 heuristics first.")

    output = pd.DataFrame(
        {
            REPORT_ID: frame[REPORT_ID],
            "heuristic_disagrees": heuristic_disagrees.to_numpy(),
            "heuristic_rules_triggered": rules if isinstance(rules, pd.Series) else pd.NA,
        }
    )
    return output.drop_duplicates(subset=[REPORT_ID], keep="first")


def load_phase1_signal(predictions_path: Path = DEFAULT_PHASE1_PREDICTIONS_PATH) -> pd.DataFrame:
    """Return Phase 1 confident-disagreement signal keyed by report id."""

    frame = _read_csv(predictions_path)
    if REPORT_ID not in frame.columns:
        raise ValueError(f"Phase 1 predictions {predictions_path} must include {REPORT_ID!r}.")

    if "classifier_flag_for_review" in frame.columns:
        phase1_disagrees = _to_bool(frame["classifier_flag_for_review"])
    elif "classifier_disagrees_with_reported_label" in frame.columns:
        phase1_disagrees = _to_bool(frame["classifier_disagrees_with_reported_label"])
    else:
        phase1_disagrees = pd.Series(False, index=frame.index)

    output = pd.DataFrame({REPORT_ID: frame[REPORT_ID], "phase1_disagrees": phase1_disagrees.to_numpy()})
    for source, target in [
        (REPORTED_AUTOMATION_LEVEL, REPORTED_AUTOMATION_LEVEL),
        (REPORTING_ENTITY, REPORTING_ENTITY),
        ("predicted_automation_level", "phase1_predicted_automation_level"),
        ("prediction_confidence", "phase1_prediction_confidence"),
    ]:
        if source in frame.columns:
            output[target] = frame[source]
    return output.drop_duplicates(subset=[REPORT_ID], keep="first")


def load_phase2_signal(comparison_path: Path = DEFAULT_COMPARISON_PATH) -> pd.DataFrame:
    """Return Phase 2 narrative-disagreement signal keyed by report id."""

    frame = _read_csv(comparison_path)
    if REPORT_ID not in frame.columns:
        raise ValueError(f"Phase 2 comparison {comparison_path} must include {REPORT_ID!r}.")

    inferred = (
        _clean_label(frame["phase2_llm_inferred_automation_level"])
        if "phase2_llm_inferred_automation_level" in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="string")
    )
    reported = (
        _clean_label(frame[REPORTED_AUTOMATION_LEVEL])
        if REPORTED_AUTOMATION_LEVEL in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="string")
    )
    status = (
        frame["phase2_inference_status"].astype("string").str.strip().str.casefold()
        if "phase2_inference_status" in frame.columns
        else pd.Series("ok", index=frame.index, dtype="string")
    )

    usable = inferred.notna() & (status.isna() | (status == "ok"))
    phase2_disagrees = usable & reported.notna() & (inferred != reported)

    output = pd.DataFrame(
        {
            REPORT_ID: frame[REPORT_ID],
            "phase2_available": usable.to_numpy(),
            "phase2_disagrees": phase2_disagrees.to_numpy(),
            "phase2_llm_inferred_automation_level": inferred.to_numpy(),
        }
    )
    if "disagreement_category" in frame.columns:
        output["phase2_disagreement_category"] = frame["disagreement_category"].to_numpy()
    return output.drop_duplicates(subset=[REPORT_ID], keep="first")


def build_consensus_queue(
    phase1: pd.DataFrame,
    heuristics: pd.DataFrame,
    phase2: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the three signals and rank records by consensus strength."""

    merged = phase1.merge(heuristics, on=REPORT_ID, how="left").merge(phase2, on=REPORT_ID, how="left")

    for column, default in [
        ("heuristic_disagrees", False),
        ("phase1_disagrees", False),
        ("phase2_disagrees", False),
        ("phase2_available", False),
    ]:
        if column not in merged.columns:
            merged[column] = default
        merged[column] = merged[column].fillna(default).astype(bool)

    merged["consensus_score"] = (
        merged["heuristic_disagrees"].astype(int)
        + merged["phase1_disagrees"].astype(int)
        + merged["phase2_disagrees"].astype(int)
    )
    merged["consensus_tier"] = merged["consensus_score"].map(TIER_BY_SCORE)
    merged["review_priority"] = merged["consensus_tier"].map(TIER_PRIORITY)
    merged["responsible_use"] = RESPONSIBLE_USE_NOTE

    if "phase1_prediction_confidence" in merged.columns:
        merged["_rank_confidence"] = pd.to_numeric(
            merged["phase1_prediction_confidence"], errors="coerce"
        ).fillna(0.0)
    else:
        merged["_rank_confidence"] = 0.0

    queue = merged[merged["consensus_score"] >= 1].copy()
    queue = queue.sort_values(
        ["consensus_score", "_rank_confidence", REPORT_ID],
        ascending=[False, False, True],
        kind="stable",
    ).drop(columns="_rank_confidence")

    output_columns = [
        REPORT_ID,
        REPORTING_ENTITY,
        REPORTED_AUTOMATION_LEVEL,
        "consensus_score",
        "consensus_tier",
        "review_priority",
        "heuristic_disagrees",
        "phase1_disagrees",
        "phase2_disagrees",
        "phase2_available",
        "phase1_predicted_automation_level",
        "phase1_prediction_confidence",
        "phase2_llm_inferred_automation_level",
        "heuristic_rules_triggered",
        "phase2_disagreement_category",
        "responsible_use",
    ]
    available = [column for column in output_columns if column in queue.columns]
    return queue.loc[:, available].reset_index(drop=True)


def summarize_queue(queue: pd.DataFrame, *, total_records: int) -> dict[str, Any]:
    """Build a JSON-serializable summary of the consensus queue."""

    tier_counts = {tier: int((queue["consensus_tier"] == tier).sum()) for tier in TIER_PRIORITY}
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_records_considered": int(total_records),
        "records_in_queue": int(len(queue)),
        "tier_counts": tier_counts,
        "records_with_phase2_evidence": int(queue["phase2_available"].sum()) if len(queue) else 0,
        "signal_totals": {
            "heuristic_disagrees": int(queue["heuristic_disagrees"].sum()) if len(queue) else 0,
            "phase1_disagrees": int(queue["phase1_disagrees"].sum()) if len(queue) else 0,
            "phase2_disagrees": int(queue["phase2_disagrees"].sum()) if len(queue) else 0,
        },
        "responsible_use": RESPONSIBLE_USE_NOTE,
    }


def plot_consensus_tiers(queue: pd.DataFrame, output_path: Path) -> None:
    """Write a small tier-count chart for local reporting."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    order = list(TIER_PRIORITY)
    counts = queue["consensus_tier"].value_counts().reindex(order, fill_value=0)
    plt.figure(figsize=(7, 4.5))
    counts.plot(kind="bar", color=["#b2182b", "#ef8a62", "#fddbc7"])
    plt.ylabel("Record count")
    plt.title("Consensus review queue by signal agreement")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_consensus(
    *,
    phase1_predictions_path: Path = DEFAULT_PHASE1_PREDICTIONS_PATH,
    heuristic_scored_path: Path = DEFAULT_HEURISTIC_SCORED_PATH,
    heuristic_flags_path: Path = DEFAULT_HEURISTIC_FLAGS_PATH,
    comparison_path: Path = DEFAULT_COMPARISON_PATH,
    queue_path: Path = DEFAULT_QUEUE_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    chart_path: Path | None = DEFAULT_CHART_PATH,
) -> dict[str, Any]:
    """Build the consensus review queue and write CSV/JSON/chart artifacts."""

    phase1 = load_phase1_signal(phase1_predictions_path)
    heuristics = load_heuristic_flags(heuristic_scored_path, heuristic_flags_path)
    phase2 = load_phase2_signal(comparison_path)

    queue = build_consensus_queue(phase1, heuristics, phase2)
    summary = summarize_queue(queue, total_records=len(phase1))

    queue_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(queue_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if chart_path is not None and len(queue):
        plot_consensus_tiers(queue, chart_path)
        summary["chart_path"] = str(chart_path)
    summary["queue_path"] = str(queue_path)
    summary["summary_path"] = str(summary_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for ``python -m src.consensus``."""

    parser = argparse.ArgumentParser(description="Fuse Phase 0/1/2 signals into a ranked consensus review queue.")
    parser.add_argument("--phase1-predictions", type=Path, default=DEFAULT_PHASE1_PREDICTIONS_PATH)
    parser.add_argument("--heuristic-scored", type=Path, default=DEFAULT_HEURISTIC_SCORED_PATH)
    parser.add_argument("--heuristic-flags", type=Path, default=DEFAULT_HEURISTIC_FLAGS_PATH)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON_PATH)
    parser.add_argument("--queue-output", type=Path, default=DEFAULT_QUEUE_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--chart-output", type=Path, default=DEFAULT_CHART_PATH)
    parser.add_argument("--no-chart", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_consensus(
        phase1_predictions_path=args.phase1_predictions,
        heuristic_scored_path=args.heuristic_scored,
        heuristic_flags_path=args.heuristic_flags,
        comparison_path=args.comparison,
        queue_path=args.queue_output,
        summary_path=args.summary_output,
        chart_path=None if args.no_chart else args.chart_output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
