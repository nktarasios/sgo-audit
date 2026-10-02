"""Smoke tests for the showcase data builder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_showcase_data.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_showcase_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_snapshot_has_expected_shape():
    module = _load_builder()
    snapshot = module.build_snapshot()

    assert snapshot["meta"]["data_vintage"] == "Archive-2021-2025"
    assert snapshot["corpus"]["combined"] == 4589
    assert snapshot["signals"]["consensus_queue"] == len(snapshot["queue"])
    assert snapshot["signals"]["classifier_flags"] == len(snapshot["classifier_flags"])
    assert snapshot["operating_point"]["model"] == "lightgbm"
    assert "heuristic_flags" in snapshot["validation"]["overlaps"]
    assert len(snapshot["tools_plain"]) == 4
    assert len(snapshot["demo_cases"]) >= 2
    assert snapshot["demo_cases"][-1]["kind"] == "agreement"


def test_build_writes_data_js(tmp_path, monkeypatch):
    module = _load_builder()
    out = tmp_path / "data.js"
    monkeypatch.setattr(module, "OUT", out)
    module.main()

    text = out.read_text(encoding="utf-8")
    assert text.startswith("window.SGO_SHOWCASE = ")
    payload = text.removeprefix("window.SGO_SHOWCASE = ").rstrip(";\n")
    data = json.loads(payload)
    assert data["meta"]["title"] == "SGO-Audit"
