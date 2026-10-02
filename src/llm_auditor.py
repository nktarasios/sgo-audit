"""Phase 2 local LLM narrative auditor for SGO classification review.

This module asks a local Ollama-served open-weight model one narrow question
about each non-redacted crash narrative: whether the narrative alone most
closely describes ADS (L3+) operation or Level 2 ADAS requiring an attentive
human driver. Outputs are review signals only; they are not findings of
misclassification, fault, severity, or manufacturer safety performance.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .fields import (
    ADAS_LEVEL,
    ADS_LEVEL,
    NARRATIVE,
    REPORT_ID,
    REPORTED_AUTOMATION_LEVEL,
    REPORTING_ENTITY,
    redaction_flag_column,
)
from .ingest import is_redaction_marker

DEFAULT_INPUT_PATH = Path("data/processed/combined_audit_frame.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/llm_audit_results.csv")
DEFAULT_SAMPLE_SIZE = 50
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_TRANSFORMERS_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TIMEOUT_SECONDS = 120
PROMPT_VERSION = "phase2_narrative_label_v1"
FALLBACK_BACKEND = "heuristic_fallback"
OLLAMA_BACKEND = "ollama"
TRANSFORMERS_BACKEND = "transformers"
RESPONSIBLE_USE_NOTE = (
    "Flags are for human review only; they are not confirmed misclassifications "
    "or safety-rate claims."
)

_TRANSFORMERS_CACHE: dict[str, Any] = {}


class OllamaUnavailable(RuntimeError):
    """Raised when the local Ollama HTTP endpoint cannot serve a request."""


class LLMParseError(ValueError):
    """Raised when a model response does not contain a supported label."""


@dataclass(frozen=True)
class ParsedInference:
    """Normalized model inference extracted from a narrative-audit response."""

    label: str
    rationale: str


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _coerce_bool(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    return text in {"1", "true", "t", "yes", "y"}


def has_usable_narrative(record: Mapping[str, Any]) -> bool:
    """Return True when a record has non-empty, non-redacted narrative text."""

    narrative = _text(record.get(NARRATIVE))
    if not narrative or is_redaction_marker(narrative):
        return False
    redaction_column = redaction_flag_column(NARRATIVE)
    if _coerce_bool(record.get(redaction_column)):
        return False
    return True


def build_prompt(narrative: str) -> str:
    """Build the constrained, single-purpose prompt for one narrative."""

    return (
        "You are auditing a NHTSA SGO crash report narrative for automation-level "
        "classification consistency.\n\n"
        "Task: Based on the narrative alone, was the vehicle most likely operating "
        "as an automated driving system (ADS, SAE L3+) or as a Level 2 ADAS "
        "system requiring an attentive human driver?\n\n"
        "Do not assess crash severity, fault, causation, or safety performance. "
        "Do not use any manufacturer-level assumptions.\n\n"
        "Return only valid JSON with exactly these keys. The label value must be "
        'exactly "ADS" or "Level 2 ADAS":\n'
        '{"label":"ADS","rationale":"one sentence citing exact phrase(s) from the narrative"}\n\n'
        f"Narrative:\n{narrative}"
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    for candidate in (stripped,):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if isinstance(value, dict):
            return value
    return None


def normalize_label(value: object) -> str | None:
    """Normalize varied model label text into the two Phase 2 labels."""

    text = _text(value)
    if not text:
        return None
    normalized = " ".join(text.casefold().replace("_", " ").split())
    adas_patterns = (
        r"\blevel\s*2\b",
        r"\bl2\b",
        r"\badas\b",
        r"\bdriver[-\s]?assist(?:ance)?\b",
        r"\battentive human driver\b",
    )
    ads_patterns = (
        r"\bads\b",
        r"\bautomated driving system\b",
        r"\bautonomous driving system\b",
        r"\blevel\s*[3-5]\b",
        r"\bl[3-5]\b",
        r"\bl3\+\b",
        r"\bfully automated\b",
        r"\bdriverless\b",
    )
    if any(re.search(pattern, normalized) for pattern in adas_patterns):
        return ADAS_LEVEL
    if any(re.search(pattern, normalized) for pattern in ads_patterns):
        return ADS_LEVEL
    return None


def parse_llm_output(text: str) -> ParsedInference:
    """Parse JSON or simple label/rationale text returned by the model."""

    payload = _extract_json_object(text)
    if payload is not None:
        label = normalize_label(
            payload.get("label")
            or payload.get("classification")
            or payload.get("automation_level")
            or payload.get("answer")
        )
        rationale = _text(payload.get("rationale") or payload.get("justification") or payload.get("reason"))
        if label and rationale:
            return ParsedInference(label=label, rationale=rationale)

    label_match = re.search(
        r"(?:^|\n)\s*(?:label|classification|answer)\s*[:\-]\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )
    label = normalize_label(label_match.group(1) if label_match else text)
    rationale_match = re.search(
        r"(?:^|\n)\s*(?:rationale|justification|reason)\s*[:\-]\s*(.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    rationale = _text(rationale_match.group(1) if rationale_match else "")
    if label and not rationale:
        # Accept a label-only response and keep the remainder as light rationale.
        remainder = text
        if label_match:
            remainder = text[label_match.end() :]
        rationale = _text(remainder) or "Model returned a label without an explicit rationale line."
    if label and rationale:
        return ParsedInference(label=label, rationale=rationale.splitlines()[0].strip())
    raise LLMParseError("Model response did not contain a parseable label and rationale.")


def call_ollama_generate(
    narrative: str,
    *,
    model: str,
    url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Call Ollama's HTTP generate API and return the generated text."""

    payload = {
        "model": model,
        "prompt": build_prompt(narrative),
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 140,
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OllamaUnavailable(f"Ollama request failed: {exc}") from exc

    try:
        response_payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OllamaUnavailable("Ollama returned non-JSON response.") from exc
    if response_payload.get("error"):
        raise OllamaUnavailable(f"Ollama returned an error: {response_payload['error']}")
    generated = response_payload.get("response")
    if generated is None and isinstance(response_payload.get("message"), dict):
        generated = response_payload["message"].get("content")
    if not isinstance(generated, str) or not generated.strip():
        raise OllamaUnavailable("Ollama response did not include generated text.")
    return generated


def _load_transformers_model(model_name: str) -> tuple[Any, Any]:
    """Load and cache a local Hugging Face causal LM for CPU inference."""

    cached = _TRANSFORMERS_CACHE.get(model_name)
    if cached is not None:
        return cached

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise OllamaUnavailable(
            "transformers/torch are not installed for the transformers backend."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()
    _TRANSFORMERS_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


def call_transformers_generate(narrative: str, *, model: str) -> str:
    """Generate a Phase 2 label/rationale using a local transformers model."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise OllamaUnavailable("torch is required for the transformers backend.") from exc

    tokenizer, causal_model = _load_transformers_model(model)
    prompt = build_prompt(narrative)
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:  # pragma: no cover
        text = prompt
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output_ids = causal_model.generate(
            **inputs,
            max_new_tokens=140,
            do_sample=False,
        )
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
    if not generated.strip():
        raise OllamaUnavailable("transformers backend returned empty text.")
    return generated


ADS_KEYWORDS = (
    "level 4",
    "level 3",
    "l4 ads",
    "l3 ads",
    "ads was engaged",
    "ads engaged",
    "autonomous mode",
    "autonomous vehicle",
    "driverless",
    "robotaxi",
    "no driver",
    "remote operator",
)

ADAS_KEYWORDS = (
    "level 2",
    "l2",
    "adas",
    "autopilot",
    "driver assist",
    "driver-assist",
    "driver assistance",
    "driver took over",
    "driver takeover",
    "hands on wheel",
    "attentive driver",
)


def _matching_phrases(narrative: str, phrases: tuple[str, ...]) -> list[str]:
    lowered = narrative.casefold()
    return [phrase for phrase in phrases if phrase in lowered]


def heuristic_fallback_inference(narrative: str) -> ParsedInference:
    """Return a simple keyword-only fallback inference for smoke testing."""

    ads_matches = _matching_phrases(narrative, ADS_KEYWORDS)
    adas_matches = _matching_phrases(narrative, ADAS_KEYWORDS)
    if len(ads_matches) >= len(adas_matches) and ads_matches:
        phrase = ads_matches[0]
        return ParsedInference(
            label=ADS_LEVEL,
            rationale=f'Keyword fallback matched narrative phrase "{phrase}".',
        )
    if adas_matches:
        phrase = adas_matches[0]
        return ParsedInference(
            label=ADAS_LEVEL,
            rationale=f'Keyword fallback matched narrative phrase "{phrase}".',
        )
    return ParsedInference(
        label=ADAS_LEVEL,
        rationale="Keyword fallback found no ADS-specific phrase; defaulted to the supervised Level 2 review label.",
    )


def audit_record(
    record: Mapping[str, Any],
    *,
    model: str,
    url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    dry_run: bool = False,
    allow_fallback: bool = True,
    backend: str = "auto",
) -> dict[str, Any]:
    """Audit one already-filtered record and return a CSV-ready result row."""

    narrative = _text(record.get(NARRATIVE))
    selected_backend = backend
    raw_response = ""
    status = "ok"
    try:
        if dry_run:
            raise OllamaUnavailable("dry-run requested")
        if selected_backend == "auto":
            # Preserve the original Phase 2 contract: Ollama first, then labeled
            # keyword fallback. Use --backend transformers explicitly for the
            # Hugging Face CPU path when Ollama's runner is unavailable.
            selected_backend = OLLAMA_BACKEND
        if selected_backend == OLLAMA_BACKEND:
            raw_response = call_ollama_generate(
                narrative, model=model, url=url, timeout=timeout
            )
        elif selected_backend == TRANSFORMERS_BACKEND:
            try:
                raw_response = call_transformers_generate(narrative, model=model)
            except Exception as transformers_exc:  # noqa: BLE001 - optional backend
                raise OllamaUnavailable(
                    f"transformers backend failed: {transformers_exc}"
                ) from transformers_exc
        else:
            raise ValueError(f"Unsupported backend: {selected_backend}")
        parsed = parse_llm_output(raw_response)
    except LLMParseError as exc:
        parsed = ParsedInference(label="", rationale=f"Parse failure: {exc}")
        status = "parse_error"
    except OllamaUnavailable as exc:
        if not allow_fallback:
            raise
        selected_backend = FALLBACK_BACKEND
        status = "ok"
        raw_response = f"fallback_reason={exc}"
        parsed = heuristic_fallback_inference(narrative)

    model_name = (
        "keyword-heuristic-fallback"
        if selected_backend == FALLBACK_BACKEND
        else model
    )
    return {
        REPORT_ID: _text(record.get(REPORT_ID)),
        REPORTING_ENTITY: _text(record.get(REPORTING_ENTITY)),
        REPORTED_AUTOMATION_LEVEL: _text(record.get(REPORTED_AUTOMATION_LEVEL)),
        "llm_inferred_automation_level": parsed.label,
        "llm_rationale": parsed.rationale,
        "inference_backend": selected_backend,
        "inference_status": status,
        "model_name": model_name,
        "prompt_version": PROMPT_VERSION,
        "raw_model_response": raw_response,
        "responsible_use": RESPONSIBLE_USE_NOTE,
    }


def select_narrative_records(frame: pd.DataFrame, *, sample: int | None) -> pd.DataFrame:
    """Return first N rows with usable narratives, preserving input order."""

    usable_mask = frame.apply(has_usable_narrative, axis=1)
    selected = frame.loc[usable_mask].copy()
    if sample is not None and sample > 0:
        selected = selected.head(sample)
    return selected.reset_index(drop=True)


def audit_frame(
    frame: pd.DataFrame,
    *,
    sample: int | None = DEFAULT_SAMPLE_SIZE,
    model: str = DEFAULT_OLLAMA_MODEL,
    url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    dry_run: bool = False,
    allow_fallback: bool = True,
    backend: str = "auto",
) -> pd.DataFrame:
    """Audit sampled narratives and return Phase 2 inference rows."""

    selected = select_narrative_records(frame, sample=sample)
    rows = [
        audit_record(
            record,
            model=model,
            url=url,
            timeout=timeout,
            dry_run=dry_run,
            allow_fallback=allow_fallback,
            backend=backend,
        )
        for record in selected.to_dict(orient="records")
    ]
    return pd.DataFrame(rows)


def run_audit(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    sample: int | None = DEFAULT_SAMPLE_SIZE,
    model: str = DEFAULT_OLLAMA_MODEL,
    url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    dry_run: bool = False,
    allow_fallback: bool = True,
    backend: str = "auto",
) -> dict[str, Any]:
    """Load input records, write the Phase 2 audit CSV, and return summary."""

    frame = pd.read_csv(input_path, dtype="string", keep_default_na=False, na_values=[])
    results = audit_frame(
        frame,
        sample=sample,
        model=model,
        url=url,
        timeout=timeout,
        dry_run=dry_run,
        allow_fallback=allow_fallback,
        backend=backend,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    backend_counts = (
        results["inference_backend"].value_counts(dropna=False).to_dict()
        if "inference_backend" in results.columns
        else {}
    )
    status_counts = (
        results["inference_status"].value_counts(dropna=False).to_dict()
        if "inference_status" in results.columns
        else {}
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "requested_sample": sample,
        "audited_rows": int(len(results)),
        "inference_backend_counts": {str(key): int(value) for key, value in backend_counts.items()},
        "inference_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "ollama_url": url,
        "model": model,
        "backend": backend,
        "model_selection_note": (
            "Prefer local Ollama on GPU when available. If Ollama's runner fails "
            "(e.g. CPU segfault in restricted VMs), use --backend transformers with "
            f"{DEFAULT_TRANSFORMERS_MODEL} or another small open-weight instruct model."
        ),
        "responsible_use": RESPONSIBLE_USE_NOTE,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for ``python -m src.llm_auditor``."""

    parser = argparse.ArgumentParser(description="Run Phase 2 local Ollama narrative audit.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Audit the first N non-redacted narratives. Use 0 for all usable narratives.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        help=(
            "Model name for the selected backend. For Ollama, defaults to "
            "OLLAMA_MODEL or llama3.1:8b. For transformers, pass a Hugging Face "
            f"model id such as {DEFAULT_TRANSFORMERS_MODEL}."
        ),
    )
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL))
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--backend",
        choices=["auto", "ollama", "transformers"],
        default=os.environ.get("SGO_LLM_BACKEND", "auto"),
        help=(
            "Inference backend. 'auto' tries Ollama then transformers. "
            "'transformers' is the reliable CPU path when Ollama's runner crashes."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Ollama/transformers and use the clearly labeled keyword heuristic fallback.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Fail instead of using heuristic_fallback when local LLM backends are unavailable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    sample = None if args.sample <= 0 else args.sample
    model = args.model
    if args.backend == TRANSFORMERS_BACKEND and model == DEFAULT_OLLAMA_MODEL:
        model = os.environ.get("TRANSFORMERS_MODEL", DEFAULT_TRANSFORMERS_MODEL)
    summary = run_audit(
        input_path=args.input,
        output_path=args.output,
        sample=sample,
        model=model,
        url=args.ollama_url,
        timeout=args.timeout,
        dry_run=args.dry_run,
        allow_fallback=not args.no_fallback,
        backend=args.backend,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
