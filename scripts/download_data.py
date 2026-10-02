"""Download public NHTSA SGO 2021-01 raw data files.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --current

Defaults download the archived 2021-2025 snapshot used for Phase 0. The
``--current`` option downloads the current third-amended CSV paths documented by
NHTSA for users who intentionally want the latest live files instead.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ARCHIVE_BASE_URL = "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/Archive-2021-2025"
CURRENT_BASE_URL = "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01"

ARCHIVE_URLS = {
    "SGO-2021-01_Incident_Reports_ADS.csv": f"{ARCHIVE_BASE_URL}/SGO-2021-01_Incident_Reports_ADS.csv",
    "SGO-2021-01_Incident_Reports_ADAS.csv": f"{ARCHIVE_BASE_URL}/SGO-2021-01_Incident_Reports_ADAS.csv",
    "SGO-2021-01_Data_Element_Definitions.pdf": f"{ARCHIVE_BASE_URL}/SGO-2021-01_Data_Element_Definitions.pdf",
}

CURRENT_URLS = {
    "SGO-2021-01_Incident_Reports_ADS.csv": f"{CURRENT_BASE_URL}/SGO-2021-01_Incident_Reports_ADS.csv",
    "SGO-2021-01_Incident_Reports_ADAS.csv": f"{CURRENT_BASE_URL}/SGO-2021-01_Incident_Reports_ADAS.csv",
}


def download_file(url: str, destination: Path, overwrite: bool = False) -> None:
    """Download one URL to ``destination``."""

    if destination.exists() and not overwrite:
        print(f"exists, skipping: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    print(f"downloading {url} -> {destination}")
    with urllib.request.urlopen(url) as response, temporary_path.open("wb") as output:
        output.write(response.read())
    temporary_path.replace(destination)


def write_vintage_file(output_dir: Path, *, current: bool) -> None:
    label = "current-third-amended" if current else "Archive-2021-2025"
    base_url = CURRENT_BASE_URL if current else f"{ARCHIVE_BASE_URL}/"
    timestamp = datetime.now(timezone.utc).isoformat()
    (output_dir / "DATA_VINTAGE.txt").write_text(
        f"DATA_VINTAGE={label}\nDownloaded from {base_url}\n{timestamp}\n",
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download public NHTSA SGO raw data.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--current",
        action="store_true",
        help="Download current third-amended ADS/ADAS CSVs instead of Archive-2021-2025.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    urls = CURRENT_URLS if args.current else ARCHIVE_URLS
    for filename, url in urls.items():
        download_file(url, args.output_dir / filename, overwrite=args.overwrite)
    write_vintage_file(args.output_dir, current=args.current)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
