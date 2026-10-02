"""Phase 0 explainable classification-consistency heuristics.

Usage:
    python -m src.heuristics
    python -m src.heuristics --input data/processed/combined_audit_frame.csv

Each rule is a named function returning ``(flag: bool, reason: str)``. These
flags are review signals only; they are not findings of misclassification and
make no safety-rate claims.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .fields import (
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

RuleResult = tuple[bool, str]
RuleFunction = Callable[[Mapping[str, Any]], RuleResult]

RULE_SEPARATOR = "; "

LEVEL2_NARRATIVE_PATTERN = re.compile(
    r"\b("
    r"autopilot|"
    r"driver\s+(?:takeover|took\s+over|intervened|intervention|disengaged)|"
    r"hands?\s+on\s+(?:the\s+)?wheel|"
    r"steering\s+wheel\s+hands|"
    r"supervised\s+(?:driver\s+)?assist|"
    r"level\s*2|"
    r"driver\s+assist(?:ance)?"
    r")\b",
    flags=re.IGNORECASE,
)


def _text(record: Mapping[str, Any], column: str) -> str:
    value = record.get(column)
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalized(record: Mapping[str, Any], column: str) -> str:
    return " ".join(_text(record, column).casefold().split())


def _reported_level(record: Mapping[str, Any]) -> str:
    return _text(record, REPORTED_AUTOMATION_LEVEL)


def ads_file_but_adas_engaged(record: Mapping[str, Any]) -> RuleResult:
    """Flag ADS-source records whose engagement field says ADAS."""

    if _reported_level(record) == ADS_LEVEL and _normalized(record, AUTOMATION_SYSTEM_ENGAGED) == "adas":
        return True, "ADS source file reports Automation System Engaged? as ADAS"
    return False, ""


def adas_file_but_ads_engaged(record: Mapping[str, Any]) -> RuleResult:
    """Flag ADAS-source records whose engagement field says ADS."""

    if _reported_level(record) == ADAS_LEVEL and _normalized(record, AUTOMATION_SYSTEM_ENGAGED) == "ads":
        return True, "ADAS source file reports Automation System Engaged? as ADS"
    return False, ""


def ads_label_but_consumer_driver(record: Mapping[str, Any]) -> RuleResult:
    """Flag ADS-labeled records with a consumer driver/operator pattern."""

    driver_type = _normalized(record, DRIVER_OPERATOR_TYPE)
    if _reported_level(record) == ADS_LEVEL and driver_type == "consumer":
        return True, "ADS-labeled record lists Driver / Operator Type as Consumer"
    return False, ""


def adas_label_but_no_driver(record: Mapping[str, Any]) -> RuleResult:
    """Flag ADAS-labeled records with a no-driver operator pattern."""

    driver_type = _normalized(record, DRIVER_OPERATOR_TYPE)
    if _reported_level(record) == ADAS_LEVEL and driver_type in {"none", "no driver"}:
        return True, "ADAS-labeled record lists Driver / Operator Type as None"
    return False, ""


def ads_narrative_level2_language(record: Mapping[str, Any]) -> RuleResult:
    """Flag narrow Level-2-supervision language in ADS-labeled narratives."""

    narrative = _text(record, NARRATIVE)
    if _reported_level(record) == ADS_LEVEL and narrative and LEVEL2_NARRATIVE_PATTERN.search(narrative):
        return True, "ADS-labeled narrative contains narrow Level-2 driver-supervision language"
    return False, ""


RULES: tuple[RuleFunction, ...] = (
    ads_file_but_adas_engaged,
    adas_file_but_ads_engaged,
    ads_label_but_consumer_driver,
    adas_label_but_no_driver,
    ads_narrative_level2_language,
)


def evaluate_record(record: Mapping[str, Any], rules: tuple[RuleFunction, ...] = RULES) -> dict[str, Any]:
    """Evaluate all heuristic rules for one record."""

    triggered: list[str] = []
    reasons: list[str] = []
    rule_flags: dict[str, bool] = {}

    for rule in rules:
        flag, reason = rule(record)
        rule_flags[rule.__name__] = bool(flag)
        if flag:
            triggered.append(rule.__name__)
            reasons.append(reason)

    return {
        **rule_flags,
        "any_heuristic_flag": bool(triggered),
        "heuristic_flag_count": len(triggered),
        "rules_triggered": RULE_SEPARATOR.join(triggered),
        "reasons": RULE_SEPARATOR.join(reasons),
    }


def score_records(frame: pd.DataFrame, rules: tuple[RuleFunction, ...] = RULES) -> pd.DataFrame:
    """Return the input records with heuristic score columns appended."""

    evaluations = [evaluate_record(row, rules) for row in frame.to_dict(orient="records")]
    scores = pd.DataFrame(evaluations, index=frame.index)
    return pd.concat([frame.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)


def flags_for_review(scored_frame: pd.DataFrame) -> pd.DataFrame:
    """Return the compact flagged-record review frame."""

    columns = [
        REPORT_ID,
        REPORTING_ENTITY,
        REPORT_YEAR,
        REPORT_MONTH,
        SOURCE_FILE,
        REPORTED_AUTOMATION_LEVEL,
        "heuristic_flag_count",
        "rules_triggered",
        "reasons",
    ]
    available_columns = [column for column in columns if column in scored_frame.columns]
    if "any_heuristic_flag" not in scored_frame.columns:
        raise ValueError("scored_frame must include any_heuristic_flag")
    return scored_frame.loc[scored_frame["any_heuristic_flag"].fillna(False), available_columns].copy()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply Phase 0 SGO consistency heuristics.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/combined_audit_frame.csv"))
    parser.add_argument("--scored-output", type=Path, default=Path("data/processed/heuristic_scored_records.csv"))
    parser.add_argument("--flags-output", type=Path, default=Path("data/processed/heuristic_flags.csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.input, dtype="string", keep_default_na=False, na_values=[])
    scored = score_records(frame)
    flags = flags_for_review(scored)

    args.scored_output.parent.mkdir(parents=True, exist_ok=True)
    args.flags_output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.scored_output, index=False)
    flags.to_csv(args.flags_output, index=False)

    rule_counts = {rule.__name__: int(scored[rule.__name__].sum()) for rule in RULES}
    print(
        json.dumps(
            {
                "records_scored": int(len(scored)),
                "records_flagged": int(scored["any_heuristic_flag"].sum()),
                "rule_counts": rule_counts,
                "scored_output": str(args.scored_output),
                "flags_output": str(args.flags_output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
