"""Phase 0 ingestion for NHTSA SGO ADAS/ADS incident CSVs.

Usage:
    python -m src.ingest
    python -m src.ingest --raw-dir data/raw --processed-dir data/processed

The loader keeps ADAS and ADS files separate, derives
``reported_automation_level`` from the source file, preserves
``Automation System Engaged?``, converts NHTSA redaction markers to missing
values, and writes cleaned CSVs plus run metadata. If a parquet engine is
available, parquet copies are written as a convenience.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .fields import (
    ADAS_LEVEL,
    ADS_LEVEL,
    KEY_FIELDS,
    OPTIONAL_KEY_FIELDS,
    REPORT_ID,
    REPORT_MONTH,
    REPORT_PERIOD,
    REPORT_VERSION,
    REPORT_YEAR,
    REPORTED_AUTOMATION_LEVEL,
    REQUIRED_KEY_FIELDS,
    SOURCE_FILE,
    normalize_sgo_columns,
    redaction_flag_column,
)


DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_ADS_FILENAME = "SGO-2021-01_Incident_Reports_ADS.csv"
DEFAULT_ADAS_FILENAME = "SGO-2021-01_Incident_Reports_ADAS.csv"
DEFAULT_VINTAGE_FILENAME = "DATA_VINTAGE.txt"

BRACKETED_REDACTION_PATTERN = re.compile(
    r"^\[\s*(?:"
    r"XXX+|"
    r"X{2,}(?:\s+X{2,})*|"
    r".*REDACTED.*|"
    r".*CONFIDENTIAL BUSINESS INFORMATION.*|"
    r".*PERSONALLY IDENTIFIABLE INFORMATION.*"
    r")\s*\]$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class IngestResult:
    """Cleaned frames and metadata produced by an ingestion run."""

    ads: pd.DataFrame
    adas: pd.DataFrame
    combined: pd.DataFrame
    metadata: dict


def is_redaction_marker(value: object) -> bool:
    """Return True when ``value`` is one of the public SGO redaction markers."""

    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    return bool(BRACKETED_REDACTION_PATTERN.match(text))


def read_sgo_csv(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Read a raw SGO CSV as strings, tolerating rare encoding defects."""

    csv_path = Path(path)
    attempts = [
        {"encoding": "utf-8", "encoding_errors": "strict"},
        {"encoding": "utf-8", "encoding_errors": "replace"},
        {"encoding": "cp1252", "encoding_errors": "replace"},
        {"encoding": "latin1", "encoding_errors": "replace"},
    ]
    last_error: Exception | None = None

    for options in attempts:
        try:
            frame = pd.read_csv(
                csv_path,
                dtype="string",
                keep_default_na=False,
                na_values=[],
                **options,
            )
            encoding_label = f"{options['encoding']}:{options['encoding_errors']}"
            return frame, encoding_label
        except UnicodeDecodeError as exc:
            last_error = exc

    assert last_error is not None
    raise last_error


def validate_key_fields(frame: pd.DataFrame, source: str) -> None:
    """Fail early if a required Phase 0 field is absent."""

    missing = [column for column in REQUIRED_KEY_FIELDS if column not in frame.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{source} is missing required SGO columns: {missing_text}")


def clean_redactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace redaction markers with NA and add ``was_redacted_*`` columns."""

    cleaned = frame.copy()
    original_columns = list(cleaned.columns)
    redaction_flags: dict[str, pd.Series] = {}

    for column in original_columns:
        if pd.api.types.is_string_dtype(cleaned[column]) or cleaned[column].dtype == object:
            cleaned[column] = cleaned[column].map(
                lambda value: value.strip() if isinstance(value, str) else value,
                na_action="ignore",
            )
        redacted_mask = cleaned[column].map(is_redaction_marker).astype("boolean").fillna(False)
        redaction_flags[redaction_flag_column(column)] = redacted_mask.astype(bool)
        cleaned.loc[redacted_mask, column] = pd.NA

    cleaned = cleaned.replace(r"^\s*$", pd.NA, regex=True)
    return pd.concat([cleaned, pd.DataFrame(redaction_flags, index=cleaned.index)], axis=1)


def add_report_period(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a YYYY-MM report period from Report Year/Month when available."""

    with_period = frame.copy()
    year = pd.to_numeric(with_period[REPORT_YEAR], errors="coerce")
    month = pd.to_numeric(with_period[REPORT_MONTH], errors="coerce")
    valid = year.notna() & month.between(1, 12)
    period = pd.Series(pd.NA, index=with_period.index, dtype="string")
    period.loc[valid] = [
        f"{int(year_value):04d}-{int(month_value):02d}"
        for year_value, month_value in zip(year.loc[valid], month.loc[valid])
    ]
    with_period[REPORT_PERIOD] = period
    return with_period


def keep_latest_report_versions(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest Report Version per Report ID."""

    if REPORT_ID not in frame.columns or REPORT_VERSION not in frame.columns:
        return frame

    version_number = pd.to_numeric(frame[REPORT_VERSION], errors="coerce").fillna(-1)
    latest = (
        frame.assign(_report_version_number=version_number, _input_order=range(len(frame)))
        .sort_values([REPORT_ID, "_report_version_number", "_input_order"], kind="stable")
        .drop_duplicates(subset=[REPORT_ID], keep="last")
        .drop(columns=["_report_version_number", "_input_order"])
        .sort_index(kind="stable")
        .reset_index(drop=True)
    )
    return latest


def clean_sgo_frame(
    frame: pd.DataFrame,
    *,
    source_file: str,
    reported_automation_level: str,
    keep_latest_versions: bool = True,
) -> pd.DataFrame:
    """Clean one raw SGO frame without merging ADAS and ADS sources."""

    normalized = normalize_sgo_columns(frame)
    validate_key_fields(normalized, source_file)
    cleaned = clean_redactions(normalized)
    cleaned[SOURCE_FILE] = source_file
    cleaned[REPORTED_AUTOMATION_LEVEL] = reported_automation_level
    cleaned = add_report_period(cleaned)
    if keep_latest_versions:
        cleaned = keep_latest_report_versions(cleaned)
    return cleaned


def read_data_vintage(vintage_path: str | Path | None, input_paths: Iterable[Path]) -> dict:
    """Collect run-level vintage metadata from DATA_VINTAGE.txt or mtimes."""

    metadata: dict[str, object] = {}
    if vintage_path:
        path = Path(vintage_path)
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            metadata["data_vintage_text"] = text
            for line in text.splitlines():
                if line.startswith("DATA_VINTAGE="):
                    metadata["data_vintage"] = line.split("=", 1)[1].strip()
                    break
            metadata["data_vintage_file"] = str(path)
            metadata["data_vintage_file_mtime_utc"] = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat()

    if "data_vintage" not in metadata:
        newest_mtime = max((path.stat().st_mtime for path in input_paths if path.exists()), default=None)
        if newest_mtime is not None:
            metadata["data_vintage"] = "file-mtime"
            metadata["latest_input_file_mtime_utc"] = datetime.fromtimestamp(
                newest_mtime, tz=timezone.utc
            ).isoformat()
        else:
            metadata["data_vintage"] = "unknown"
    return metadata


def _write_frame(frame: pd.DataFrame, path_without_suffix: Path) -> list[str]:
    """Write CSV and, when supported, parquet output for a cleaned frame."""

    written: list[str] = []
    csv_path = path_without_suffix.with_suffix(".csv")
    frame.to_csv(csv_path, index=False)
    written.append(str(csv_path))

    try:
        parquet_path = path_without_suffix.with_suffix(".parquet")
        frame.to_parquet(parquet_path, index=False)
        written.append(str(parquet_path))
    except Exception as exc:  # pragma: no cover - depends on optional parquet engine
        written.append(f"parquet_skipped:{type(exc).__name__}:{exc}")

    return written


def ingest_files(
    *,
    ads_path: str | Path,
    adas_path: str | Path,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    vintage_path: str | Path | None = None,
    keep_latest_versions: bool = True,
) -> IngestResult:
    """Load, clean, and persist ADS and ADAS raw CSVs."""

    ads_file = Path(ads_path)
    adas_file = Path(adas_path)
    output_dir = Path(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_ads, ads_encoding = read_sgo_csv(ads_file)
    raw_adas, adas_encoding = read_sgo_csv(adas_file)

    ads = clean_sgo_frame(
        raw_ads,
        source_file=ads_file.name,
        reported_automation_level=ADS_LEVEL,
        keep_latest_versions=keep_latest_versions,
    )
    adas = clean_sgo_frame(
        raw_adas,
        source_file=adas_file.name,
        reported_automation_level=ADAS_LEVEL,
        keep_latest_versions=keep_latest_versions,
    )
    combined = pd.concat([ads, adas], ignore_index=True, sort=False)

    written_outputs = {
        "ads": _write_frame(ads, output_dir / "ads_clean"),
        "adas": _write_frame(adas, output_dir / "adas_clean"),
        "combined": _write_frame(combined, output_dir / "combined_audit_frame"),
    }

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "keep_latest_report_version_per_report_id": keep_latest_versions,
        "inputs": {
            "ads": {
                "path": str(ads_file),
                "source_file": ads_file.name,
                "encoding_used": ads_encoding,
                "rows_raw": int(len(raw_ads)),
                "rows_clean": int(len(ads)),
                "columns_raw": int(len(raw_ads.columns)),
                "size_bytes": ads_file.stat().st_size if ads_file.exists() else None,
                "mtime_utc": datetime.fromtimestamp(ads_file.stat().st_mtime, tz=timezone.utc).isoformat()
                if ads_file.exists()
                else None,
            },
            "adas": {
                "path": str(adas_file),
                "source_file": adas_file.name,
                "encoding_used": adas_encoding,
                "rows_raw": int(len(raw_adas)),
                "rows_clean": int(len(adas)),
                "columns_raw": int(len(raw_adas.columns)),
                "size_bytes": adas_file.stat().st_size if adas_file.exists() else None,
                "mtime_utc": datetime.fromtimestamp(adas_file.stat().st_mtime, tz=timezone.utc).isoformat()
                if adas_file.exists()
                else None,
            },
        },
        "outputs": written_outputs,
    }
    metadata.update(read_data_vintage(vintage_path, [ads_file, adas_file]))

    metadata_path = output_dir / "ingest_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)

    return IngestResult(ads=ads, adas=adas, combined=combined, metadata=metadata)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for ``python -m src.ingest``."""

    parser = argparse.ArgumentParser(description="Clean Phase 0 SGO ADAS and ADS CSVs.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--ads-path", type=Path, default=None)
    parser.add_argument("--adas-path", type=Path, default=None)
    parser.add_argument("--vintage-path", type=Path, default=None)
    parser.add_argument(
        "--keep-all-versions",
        action="store_true",
        help="Keep all Report Version rows instead of latest per Report ID.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run ingestion from the command line."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    raw_dir = args.raw_dir
    ads_path = args.ads_path or raw_dir / DEFAULT_ADS_FILENAME
    adas_path = args.adas_path or raw_dir / DEFAULT_ADAS_FILENAME
    vintage_path = args.vintage_path or raw_dir / DEFAULT_VINTAGE_FILENAME

    result = ingest_files(
        ads_path=ads_path,
        adas_path=adas_path,
        processed_dir=args.processed_dir,
        vintage_path=vintage_path,
        keep_latest_versions=not args.keep_all_versions,
    )

    print(
        json.dumps(
            {
                "ads_rows": len(result.ads),
                "adas_rows": len(result.adas),
                "combined_rows": len(result.combined),
                "metadata_path": result.metadata["metadata_path"],
                "data_vintage": result.metadata.get("data_vintage"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
