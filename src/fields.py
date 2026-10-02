"""Shared SGO column names and small normalization helpers.

Phase 0 intentionally keeps these constants close to the verified public CSV
schema so later modules do not drift into invented field names. The third
amended (current) SGO CSVs renamed/removed some archive fields; aliases below
normalize both vintages onto one canonical column set.
"""

from __future__ import annotations

import re

import pandas as pd

REPORT_ID = "Report ID"
REPORT_VERSION = "Report Version"
REPORTING_ENTITY = "Reporting Entity"
REPORT_TYPE = "Report Type"
REPORT_MONTH = "Report Month"
REPORT_YEAR = "Report Year"
REPORT_SUBMISSION_DATE = "Report Submission Date"
DRIVER_OPERATOR_TYPE = "Driver / Operator Type"
SYSTEM_VERSION = "ADAS/ADS System Version"
ADS_EQUIPPED = "ADS Equipped?"
AUTOMATION_SYSTEM_ENGAGED = "Automation System Engaged?"
OPERATING_ENTITY = "Operating Entity"
INCIDENT_DATE = "Incident Date"
CITY = "City"
STATE = "State"
ROADWAY_TYPE = "Roadway Type"
SV_PRE_CRASH_MOVEMENT = "SV Pre-Crash Movement"
SV_PRECRASH_SPEED = "SV Precrash Speed (MPH)"
CRASH_WITH = "Crash With"
HIGHEST_INJURY_SEVERITY = "Highest Injury Severity Alleged"
NARRATIVE = "Narrative"
NARRATIVE_CBI = "Narrative - CBI?"
WITHIN_ODD = "Within ODD?"
SV_ANY_AIRBAGS = "SV Any Air Bags Deployed?"
SV_WAS_TOWED = "SV Was Vehicle Towed?"
SV_ALL_BELTED = "SV Were All Passengers Belted?"

SOURCE_FILE = "source_file"
REPORTED_AUTOMATION_LEVEL = "reported_automation_level"
REPORT_PERIOD = "report_period"

ADS_LEVEL = "ADS"
ADAS_LEVEL = "Level 2 ADAS"

# Map current third-amended column names -> archive/canonical names.
COLUMN_ALIASES = {
    "Automation Feature Version": SYSTEM_VERSION,
    "Any Air Bags Deployed?": SV_ANY_AIRBAGS,
    "Was Any Vehicle Towed?": SV_WAS_TOWED,
    "Were All Passengers Belted?": SV_ALL_BELTED,
}

REQUIRED_KEY_FIELDS = [
    REPORT_ID,
    REPORT_VERSION,
    REPORTING_ENTITY,
    REPORT_TYPE,
    REPORT_MONTH,
    REPORT_YEAR,
    DRIVER_OPERATOR_TYPE,
    AUTOMATION_SYSTEM_ENGAGED,
    OPERATING_ENTITY,
    INCIDENT_DATE,
    CITY,
    STATE,
    ROADWAY_TYPE,
    SV_PRE_CRASH_MOVEMENT,
    SV_PRECRASH_SPEED,
    CRASH_WITH,
    HIGHEST_INJURY_SEVERITY,
    NARRATIVE,
    NARRATIVE_CBI,
    WITHIN_ODD,
]

# Present in Archive-2021-2025; absent or renamed in the third-amended files.
OPTIONAL_KEY_FIELDS = [
    SYSTEM_VERSION,
    ADS_EQUIPPED,
]

KEY_FIELDS = REQUIRED_KEY_FIELDS + OPTIONAL_KEY_FIELDS

REDACTION_EXAMPLES = (
    "[REDACTED, MAY CONTAIN CONFIDENTIAL BUSINESS INFORMATION]",
    "[MAY CONTAIN PERSONALLY IDENTIFIABLE INFORMATION]",
    "[XXX]",
)

_SNAKE_CASE_PATTERN = re.compile(r"[^0-9a-zA-Z]+")


def snake_case(name: str) -> str:
    """Convert a public CSV column name into a stable suffix."""

    normalized = _SNAKE_CASE_PATTERN.sub("_", name.strip().lower()).strip("_")
    return normalized or "field"


def redaction_flag_column(column_name: str) -> str:
    """Return the boolean flag column for redactions in ``column_name``."""

    return f"was_redacted_{snake_case(column_name)}"


def normalize_sgo_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename known third-amended aliases and ensure optional columns exist."""

    normalized = frame.copy()
    rename_map = {
        source: target
        for source, target in COLUMN_ALIASES.items()
        if source in normalized.columns and target not in normalized.columns
    }
    if rename_map:
        normalized = normalized.rename(columns=rename_map)
    for column in OPTIONAL_KEY_FIELDS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
            normalized[column] = normalized[column].astype("string")
    return normalized
