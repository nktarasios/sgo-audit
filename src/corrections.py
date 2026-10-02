"""Answer key of automation-level changes across SGO report versions.

Usage:
    python -m src.corrections
    python -m src.corrections --raw-dir data/raw --output results/corrections.csv

Reads the raw ADS and ADAS CSVs *before* ``keep_latest_report_versions``.
NHTSA clusters every version of a Report ID into the file of the latest
version, so a level change is visible only while older versions are kept.

On the public SGO files the version-specific level is ``Automation System
Engaged?`` (ADS, ADAS, or Unknown). ADAS is stored here as ``Level 2 ADAS``
so the answer key uses the same labels as the rest of the pipeline. When that
column does not encode those levels, the source-file label
(``reported_automation_level``) is compared instead.

Each output row is a documented difference between consecutive versions. It is
a review input, not a finding of misclassification, and not a safety rate or
a manufacturer ranking.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import pairwise
from pathlib import Path

import pandas as pd

from .fields import (
    ADAS_LEVEL,
    ADS_LEVEL,
    AUTOMATION_SYSTEM_ENGAGED,
    CRASH_WITH,
    DRIVER_OPERATOR_TYPE,
    HIGHEST_INJURY_SEVERITY,
    INCIDENT_DATE,
    REPORT_ID,
    REPORT_PERIOD,
    REPORT_SUBMISSION_DATE,
    REPORT_VERSION,
    REPORTED_AUTOMATION_LEVEL,
    REPORTING_ENTITY,
    ROADWAY_TYPE,
    SOURCE_FILE,
    SV_PRE_CRASH_MOVEMENT,
    SV_PRECRASH_SPEED,
    WITHIN_ODD,
)
from .ingest import (
    DEFAULT_ADAS_FILENAME,
    DEFAULT_ADS_FILENAME,
    DEFAULT_RAW_DIR,
    clean_sgo_frame,
    read_sgo_csv,
)

DEFAULT_OUTPUT = Path("results/corrections.csv")

# Structured fields the Phase 1 classifier uses, operator type first.
# Kept aligned with BASE_CATEGORICAL_FEATURES / BASE_NUMERIC_FEATURES in
# src/classifier.py. Listed here so this module stays runnable without
# importing the model stack.
CONTEXT_FIELDS = (
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
    SV_PRECRASH_SPEED,
    "Posted Speed Limit (MPH)",
)

# Identity fields recorded when they change, after the classifier fields.
_IDENTITY_CONTEXT_FIELDS = (REPORTING_ENTITY, SOURCE_FILE)

LEVEL_LABELS = frozenset({ADS_LEVEL, ADAS_LEVEL})
UNKNOWN_LABEL = "Unknown"
BLANK_LABEL = "(blank)"
AUTOMATION_LEVEL_KIND = "automation_level"
OTHER_KIND = "other"
LEVEL_VOCABULARY = frozenset({ADS_LEVEL, ADAS_LEVEL, UNKNOWN_LABEL})

CORRECTION_COLUMNS = (
    REPORT_ID,
    "versions",
    "old_label",
    "new_label",
    "dates",
    "correction_kind",
    "earlier_version",
    "later_version",
    "earlier_date",
    "later_date",
    "label_source",
    SOURCE_FILE,
    REPORTING_ENTITY,
    "old_operator_type",
    "new_operator_type",
    "context_field_changes",
)

REVIEW_LANGUAGE = (
    "Rows are documented version differences for review, not findings of "
    "misclassification or safety rates."
)


def canonical_automation_label(value: object) -> str | None:
    """Map an engaged-system or source-file value onto a stable label."""

    if _is_missing(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    folded = text.casefold()
    if folded == "ads":
        return ADS_LEVEL
    if folded in {"adas", "level 2 adas"}:
        return ADAS_LEVEL
    if folded.startswith("unknown"):
        return UNKNOWN_LABEL
    return text


def engagement_field_encodes_level(frame: pd.DataFrame) -> bool:
    """True when ``Automation System Engaged?`` carries ADS, ADAS, or Unknown."""

    if AUTOMATION_SYSTEM_ENGAGED not in frame.columns:
        return False
    labels = frame[AUTOMATION_SYSTEM_ENGAGED].map(canonical_automation_label)
    return bool(labels.isin(list(LEVEL_VOCABULARY)).any())


def load_versioned_reports(ads_path: str | Path, adas_path: str | Path) -> pd.DataFrame:
    """Clean both raw CSVs and keep every Report Version."""

    ads_file = Path(ads_path)
    adas_file = Path(adas_path)
    raw_ads, _encoding = read_sgo_csv(ads_file)
    raw_adas, _encoding = read_sgo_csv(adas_file)
    ads = clean_sgo_frame(
        raw_ads,
        source_file=ads_file.name,
        reported_automation_level=ADS_LEVEL,
        keep_latest_versions=False,
    )
    adas = clean_sgo_frame(
        raw_adas,
        source_file=adas_file.name,
        reported_automation_level=ADAS_LEVEL,
        keep_latest_versions=False,
    )
    return pd.concat([ads, adas], ignore_index=True, sort=False)


def extract_corrections(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return correction rows and a count summary for ``frame``.

    ``frame`` must already include every version. Collapsing to the latest
    version first drops the earlier label this answer key is built from.
    """

    if REPORT_ID not in frame.columns or REPORT_VERSION not in frame.columns:
        raise ValueError(
            "Versioned SGO frame must include Report ID and Report Version."
        )

    encodes_level = engagement_field_encodes_level(frame)
    label_source = (
        AUTOMATION_SYSTEM_ENGAGED if encodes_level else REPORTED_AUTOMATION_LEVEL
    )
    if label_source not in frame.columns:
        raise ValueError(
            f"Versioned SGO frame is missing the automation-level field: {label_source}"
        )

    corrections = _correction_rows(
        frame, label_source=label_source, encodes_level=encodes_level
    )
    summary = summarize_corrections(corrections, label_source=label_source)
    summary["version_rows"] = len(frame)
    report_ids = frame[REPORT_ID].astype("string").str.strip()
    summary["report_ids"] = int(
        report_ids[report_ids.ne("") & report_ids.ne("<NA>")].nunique()
    )
    return corrections, summary


def summarize_corrections(
    corrections: pd.DataFrame, *, label_source: str | None = None
) -> dict:
    """Count automation-level changes separately from other field changes."""

    if corrections.empty or "correction_kind" not in corrections.columns:
        level = corrections.iloc[0:0]
        other = level
        source = label_source
    else:
        level = corrections.loc[corrections["correction_kind"] == AUTOMATION_LEVEL_KIND]
        other = corrections.loc[corrections["correction_kind"] == OTHER_KIND]
        if (
            label_source is None
            and "label_source" in corrections.columns
            and len(corrections)
        ):
            source = str(corrections["label_source"].iloc[0])
        else:
            source = label_source

    summary = {
        "label_source": source,
        "correction_rows": len(corrections),
        "automation_level_corrections": _nunique_ids(level),
        "automation_level_rows": len(level),
        "other_field_changes": _nunique_ids(other),
        "other_field_rows": len(other),
        "review_language": REVIEW_LANGUAGE,
    }
    return summary


def summarize_corrections_file(path: str | Path) -> dict:
    """Summarize a corrections CSV written by this module."""

    frame = pd.read_csv(path, dtype="string", keep_default_na=False, na_values=[])
    return summarize_corrections(frame)


def format_summary_lines(summary: dict | None) -> list[str]:
    """Markdown block for ``results/RUN_SUMMARY.md``."""

    lines = [
        "## Corrections answer key (report versions)",
        "",
        "Documented changes in the automation-level field between an earlier and a",
        "later Report Version of the same Report ID. Built from the raw CSVs before",
        "latest-version collapse. These rows are version history for review, not",
        "findings of misclassification and not safety rates.",
        "",
    ]
    if summary is None:
        lines.extend(
            [
                "- Automation-level corrections (ADS and Level 2 ADAS): `n/a`",
                "- `results/corrections.csv` was not generated for this run.",
                "",
            ]
        )
        return lines

    source = summary.get("label_source") or "n/a"
    lines.extend(
        [
            f"- Label field: `{source}`",
            (
                "- Automation-level corrections (ADS and Level 2 ADAS): "
                f"`{summary['automation_level_corrections']}` reports "
                f"(`{summary['automation_level_rows']}` rows)"
            ),
            (
                "- Other changes in that field: "
                f"`{summary['other_field_changes']}` reports "
                f"(`{summary['other_field_rows']}` rows)"
            ),
            "- Artifact: `results/corrections.csv`",
            "",
        ]
    )
    return lines


def write_corrections(corrections: pd.DataFrame, output_path: str | Path) -> Path:
    """Write the answer key CSV, including a header when there are no rows."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    corrections.to_csv(path, index=False)
    return path


def build_corrections_file(
    *,
    ads_path: str | Path,
    adas_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict:
    """Load every version, write ``output_path``, and return the summary."""

    frame = load_versioned_reports(ads_path, adas_path)
    corrections, summary = extract_corrections(frame)
    written = write_corrections(corrections, output_path)
    summary["output"] = str(written)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for ``python -m src.corrections``."""

    parser = argparse.ArgumentParser(
        description="Build the corrections answer key from SGO report versions."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--ads-path", type=Path, default=None)
    parser.add_argument("--adas-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write ``results/corrections.csv`` from the raw ADS and ADAS files."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    raw_dir = args.raw_dir
    ads_path = args.ads_path or raw_dir / DEFAULT_ADS_FILENAME
    adas_path = args.adas_path or raw_dir / DEFAULT_ADAS_FILENAME
    summary = build_corrections_file(
        ads_path=ads_path,
        adas_path=adas_path,
        output_path=args.output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _correction_rows(
    frame: pd.DataFrame, *, label_source: str, encodes_level: bool
) -> pd.DataFrame:
    work = frame.copy()
    work["_input_order"] = range(len(work))
    work["_version_number"] = pd.to_numeric(work[REPORT_VERSION], errors="coerce")
    if encodes_level:
        work["_label"] = work[AUTOMATION_SYSTEM_ENGAGED].map(canonical_automation_label)
    else:
        work["_label"] = work[REPORTED_AUTOMATION_LEVEL].map(canonical_automation_label)

    report_ids = work[REPORT_ID].map(_display)
    work = work.loc[report_ids.ne("")].copy()

    records: list[dict] = []
    for report_id, group in work.groupby(REPORT_ID, sort=True):
        ordered = group.sort_values(
            ["_version_number", "_input_order"],
            kind="stable",
            na_position="last",
        )
        rows = ordered.to_dict(orient="records")
        for earlier, later in pairwise(rows):
            old = _as_label(earlier.get("_label"))
            new = _as_label(later.get("_label"))
            if old == new:
                continue
            records.append(
                _correction_record(
                    report_id=str(report_id).strip(),
                    earlier=earlier,
                    later=later,
                    old=old,
                    new=new,
                    label_source=label_source,
                )
            )

    return pd.DataFrame.from_records(records, columns=list(CORRECTION_COLUMNS))


def _correction_record(
    *,
    report_id: str,
    earlier: dict,
    later: dict,
    old: str | None,
    new: str | None,
    label_source: str,
) -> dict:
    earlier_version = _version_token(
        earlier.get(REPORT_VERSION), earlier.get("_version_number")
    )
    later_version = _version_token(
        later.get(REPORT_VERSION), later.get("_version_number")
    )
    earlier_date = _version_date(earlier)
    later_date = _version_date(later)
    kind = (
        AUTOMATION_LEVEL_KIND
        if old in LEVEL_LABELS and new in LEVEL_LABELS and old != new
        else OTHER_KIND
    )
    return {
        REPORT_ID: report_id,
        "versions": _pair(earlier_version, later_version),
        "old_label": _label_text(old),
        "new_label": _label_text(new),
        "dates": _pair(earlier_date, later_date),
        "correction_kind": kind,
        "earlier_version": earlier_version,
        "later_version": later_version,
        "earlier_date": earlier_date,
        "later_date": later_date,
        "label_source": label_source,
        SOURCE_FILE: _display(later.get(SOURCE_FILE))
        or _display(earlier.get(SOURCE_FILE)),
        REPORTING_ENTITY: _display(later.get(REPORTING_ENTITY))
        or _display(earlier.get(REPORTING_ENTITY)),
        "old_operator_type": _display(earlier.get(DRIVER_OPERATOR_TYPE)),
        "new_operator_type": _display(later.get(DRIVER_OPERATOR_TYPE)),
        "context_field_changes": _context_field_changes(earlier, later),
    }


def _context_field_changes(earlier: dict, later: dict) -> str:
    parts: list[str] = []
    for column in (*CONTEXT_FIELDS, *_IDENTITY_CONTEXT_FIELDS):
        if column not in earlier and column not in later:
            continue
        old = _context_value(earlier.get(column))
        new = _context_value(later.get(column))
        if old != new:
            parts.append(f"{column}: {old or BLANK_LABEL} -> {new or BLANK_LABEL}")
    return " | ".join(parts)


def _version_date(row: dict) -> str:
    for column in (REPORT_SUBMISSION_DATE, REPORT_PERIOD, INCIDENT_DATE):
        text = _display(row.get(column))
        if text:
            return text
    return ""


def _version_token(raw: object, number: object) -> str:
    if not _is_missing(number):
        try:
            as_float = float(number)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            as_float = None
        if as_float is not None and math.isfinite(as_float) and as_float.is_integer():
            return str(int(as_float))
        if as_float is not None and math.isfinite(as_float):
            return format(as_float, "g")
    return _display(raw)


def _context_value(value: object) -> str:
    text = _display(value)
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return text
    if number.is_integer():
        return str(int(number))
    return format(number, "g")


def _label_text(label: str | None) -> str:
    return BLANK_LABEL if label is None else label


def _pair(left: str, right: str) -> str:
    if not left and not right:
        return ""
    return f"{left}->{right}"


def _display(value: object) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _as_label(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing)


def _nunique_ids(frame: pd.DataFrame) -> int:
    if frame.empty or REPORT_ID not in frame.columns:
        return 0
    ids = frame[REPORT_ID].astype("string").str.strip()
    return int(ids[ids.ne("")].nunique())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
