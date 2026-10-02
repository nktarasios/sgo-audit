import json
import urllib.error

import pandas as pd
import pytest

from src.fields import (
    ADAS_LEVEL,
    ADS_LEVEL,
    NARRATIVE,
    REPORT_ID,
    REPORTED_AUTOMATION_LEVEL,
    REPORTING_ENTITY,
    redaction_flag_column,
)
from src.llm_auditor import (
    FALLBACK_BACKEND,
    LLMParseError,
    audit_frame,
    call_ollama_generate,
    has_usable_narrative,
    parse_llm_output,
)


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_call_ollama_generate_parses_http_success(monkeypatch):
    def fake_urlopen(request, timeout):
        assert timeout == 7
        body = json.loads(request.data.decode("utf-8"))
        assert body["model"] == "unit-model"
        assert body["stream"] is False
        return FakeHTTPResponse({"response": '{"label":"ADS","rationale":"mentions Level 4 ADS"}'})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = call_ollama_generate("Level 4 ADS was engaged.", model="unit-model", timeout=7)

    assert "Level 4 ADS" in response


def test_parse_llm_output_accepts_json_and_rejects_unstructured_text():
    parsed = parse_llm_output(
        '{"label":"Level 2 ADAS","rationale":"The narrative says \\"driver took over\\"."}'
    )

    assert parsed.label == ADAS_LEVEL
    assert "driver took over" in parsed.rationale
    with pytest.raises(LLMParseError):
        parse_llm_output("This record is complicated.")


def test_redacted_and_empty_narratives_are_skipped():
    frame = pd.DataFrame(
        [
            {
                REPORT_ID: "R-1",
                REPORTING_ENTITY: "Example",
                REPORTED_AUTOMATION_LEVEL: ADS_LEVEL,
                NARRATIVE: "[REDACTED, MAY CONTAIN CONFIDENTIAL BUSINESS INFORMATION]",
                redaction_flag_column(NARRATIVE): True,
            },
            {
                REPORT_ID: "R-2",
                REPORTING_ENTITY: "Example",
                REPORTED_AUTOMATION_LEVEL: ADS_LEVEL,
                NARRATIVE: "",
                redaction_flag_column(NARRATIVE): False,
            },
            {
                REPORT_ID: "R-3",
                REPORTING_ENTITY: "Example",
                REPORTED_AUTOMATION_LEVEL: ADS_LEVEL,
                NARRATIVE: "The vehicle's Level 4 ADS was engaged in autonomous mode.",
                redaction_flag_column(NARRATIVE): False,
            },
        ]
    )

    assert not has_usable_narrative(frame.iloc[0].to_dict())
    assert not has_usable_narrative(frame.iloc[1].to_dict())
    results = audit_frame(frame, sample=50, model="unit-model", dry_run=True)

    assert results[REPORT_ID].tolist() == ["R-3"]
    assert results["llm_inferred_automation_level"].tolist() == [ADS_LEVEL]
    assert results["inference_backend"].tolist() == [FALLBACK_BACKEND]


def test_ollama_unavailable_uses_labeled_fallback(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    frame = pd.DataFrame(
        [
            {
                REPORT_ID: "R-1",
                REPORTING_ENTITY: "Example",
                REPORTED_AUTOMATION_LEVEL: ADS_LEVEL,
                NARRATIVE: "At impact, the Level 4 ADS was engaged in autonomous mode.",
                redaction_flag_column(NARRATIVE): False,
            }
        ]
    )

    results = audit_frame(frame, sample=1, model="unit-model")

    assert results["inference_backend"].tolist() == [FALLBACK_BACKEND]
    assert results["llm_inferred_automation_level"].tolist() == [ADS_LEVEL]
