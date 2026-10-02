# SGO-Audit

[![CI](https://github.com/nktarasios/sgo-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/nktarasios/sgo-audit/actions/workflows/ci.yml)

SGO-Audit is an independent research tool for auditing automation-level
classification consistency in the public NHTSA Standing General Order (SGO)
2021-01 crash-reporting data.

Hosted showcase: **https://nktarasios.github.io/sgo-audit/** (the findings UI
served from this repo's tracked results).

## Problem statement

The public SGO crash dataset separates incidents reported as Level 2 ADAS from
incidents reported as ADS. Those labels are self-reported by manufacturers and
operators, and NHTSA has acknowledged that records can be corrected over time.
This project asks a narrow data-quality question:

> Does the reported automation-level label look consistent with the rest of the
> public record?

SGO-Audit does not decide ground truth for any crash. It produces explainable
review flags so a human researcher, journalist, or regulator can decide what to
inspect more closely.

## Results at a glance

Headline numbers from the canonical end-to-end run on NHTSA's
**Archive-2021-2025** snapshot:

| Signal | Result |
|---|---|
| Cleaned records | 4,589 (1,899 ADS + 2,690 ADAS) |
| Phase 0 heuristic consistency flags | 37 |
| Phase 1 high-confidence model-disagreement flags | 20 (LightGBM, threshold 0.59; synthetic-flip precision/recall 0.912/0.378) |
| Phase 2 local-LLM narrative check (50 narratives) | 41 three-way agreements, 0 both-methods-against-label |
| Phase 2.5 consensus review candidates | 66 |

See [WRITEUP.md](WRITEUP.md) for the full one-page read.

## Showcase UI

A static findings UI ships in [showcase/](showcase/), powered by the tracked
[results/](results/) snapshot (canonical vintage: **Archive-2021-2025**). It
walks the problem, approach, operating point, boundaries, and proof, and
includes a **mini interactive demo** of the consensus review queue.

```bash
python scripts/build_showcase_data.py   # refresh showcase/data.js from results/
python scripts/serve_showcase.py        # http://127.0.0.1:8765/
```

Open that URL for the walkthrough: method, Archive metrics, and an
interactive consensus review queue. One-page narrative:
[WRITEUP.md](WRITEUP.md). Reproducibility notes:
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

Maintainer note: the hosted deploy requires a one-time repo setting
(Settings → Pages → Source: **GitHub Actions**); the workflow in
[.github/workflows/pages.yml](.github/workflows/pages.yml) then publishes
`showcase/` on every push to `main`.

## Quickstart

Python 3.11+ is recommended.

```bash
git clone https://github.com/nktarasios/sgo-audit.git
cd sgo-audit
pip install -r requirements.txt
python scripts/download_data.py
```

The optional Phase 2 local-LLM transformers backend has its own heavyweight
dependency set (~2-3 GB); install it only if you plan to run that backend:

```bash
pip install -r requirements-llm.txt
```

By default, the downloader fetches the archived 2021-2025 SGO snapshot used for
the tracked `results/` baseline. To intentionally use NHTSA's current
third-amended CSV paths instead:

```bash
python scripts/download_data.py --current --overwrite
```

Run the phases:

```bash
# Phase 0: clean raw ADAS/ADS CSVs.
python -m src.ingest

# Phase 0: apply explainable heuristic consistency rules.
python -m src.heuristics

# Phase 0: aggregate reporting-entity classification-consistency flag rates.
python -m src.entity_trends

# Phase 1: train/evaluate structured classifiers and write ranked review flags.
python -m src.classifier
```

Or run the whole pipeline and publish a tracked summary under `results/`:

```bash
python scripts/run_all.py --llm-backend transformers --llm-sample 50
```

Add `--current` only when you intentionally want the live third-amended CSVs
instead of the Archive baseline.

Phase 2 prefers a local Ollama server on GPU. Install and start Ollama separately,
then pull a local model such as:

```bash
ollama pull llama3.1:8b
python -m src.llm_auditor --sample 50 --backend ollama --model llama3.1:8b --no-fallback
python -m src.compare
```

If Ollama's runner is unavailable (for example on some CPU-only VMs), use the
local Hugging Face transformers backend instead (requires
`pip install -r requirements-llm.txt`):

```bash
python -m src.llm_auditor --sample 50 --backend transformers --model Qwen/Qwen2.5-1.5B-Instruct --no-fallback
python -m src.compare
```

The `--dry-run` path uses a clearly labeled keyword fallback for smoke testing
only:

```bash
python -m src.llm_auditor --sample 10 --dry-run
```

Do not interpret fallback rows as LLM audit output.

Phase 3 cross-references project flag sets against the small curated list of SGO
Report IDs named in public NHTSA ODI investigation materials:

```bash
python -m src.validate
```

The default curated reference lives at
`data/reference/nhtsa_investigation_report_ids.json` and includes IDs from
EA26002 and PE24031 resumes. Validation writes
`data/processed/validation_report.json` and `outputs/validation_note.md`. Treat
the result as directional: the named records are crashes NHTSA scrutinized, not
ground-truth labels proving automation-level misclassification.

## Consensus review queue

Phase 2.5 fuses the three independent signals, Phase 0 heuristics, the Phase 1
classifier, and the Phase 2 narrative LLM, into a single ranked queue. Each
record gets a `consensus_score` (0-3) counting how many signals disagree with
the reported label, so records where multiple methods independently disagree
rank highest:

```bash
python -m src.consensus
```

This writes `data/processed/consensus_review_queue.csv`,
`data/processed/consensus_summary.json`, and `outputs/consensus_tiers.png`.
Three-signal consensus rows are the strongest review candidates; they are still
review signals only, never confirmed misclassifications.

## Phase 1 operating point (precision-first)

Phase 1 flags high-confidence model disagreements with the reported label.
Threshold selection lives in [src/classifier.py](src/classifier.py) (`choose_operating_point` /
`calibrate_threshold_with_synthetic_flips`):

1. Sweep disagreement-confidence thresholds on synthetic label flips.
2. Keep only thresholds with synthetic precision ≥ 80%
   (`DEFAULT_TARGET_SYNTHETIC_PRECISION`).
3. Among those, pick the threshold that maximizes recall (precision-first:
   few near-certain flags over broad recall).

## Calibrated confidence (opt-in; default `none`)

`--calibration {none,sigmoid,isotonic}` can wrap each Phase 1 estimator in
`CalibratedClassifierCV` so disagreement confidence uses calibrated
probabilities. **Default remains `none`.** On the Archive-2021-2025 snapshot,
sigmoid/isotonic did not measurably improve precision at the chosen operating
point and produced a larger, lower-confidence flag set, which conflicts with
the precision-first Phase 1 priority. Use the flag when exploring calibration:

```bash
python -m src.classifier --calibration isotonic
python -m src.classifier --calibration sigmoid
```

## Data source and attribution

Source data comes from NHTSA's Standing General Order on Crash Reporting:

https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting

The default downloader uses:

- `Archive-2021-2025/SGO-2021-01_Incident_Reports_ADS.csv`
- `Archive-2021-2025/SGO-2021-01_Incident_Reports_ADAS.csv`
- `Archive-2021-2025/SGO-2021-01_Data_Element_Definitions.pdf`

**Canonical vintage for tracked `results/`:** `Archive-2021-2025`. The earlier
`--current` / `current-third-amended` run (smaller corpus, threshold 0.91,
synth P/R ≈ 0.972/0.955) is retained only as a historical comparison point in
[results/RUN_SUMMARY.md](results/RUN_SUMMARY.md). The Archive re-run (threshold 0.59, synth P/R ≈
0.912/0.378) is the baseline going forward; that P/R/threshold shift is from
the snapshot change under the same precision-first rule and `calibration=none`,
not from the calibration-default decision.

NHTSA also publishes current third-amended CSVs (`--current`). Because records
can be amended, expanded, or corrected, outputs should always be read against
the run-level `data_vintage` field in `data/processed/ingest_metadata.json`
(also copied to `results/ingest_metadata.json` and surfaced in
[results/RUN_SUMMARY.md](results/RUN_SUMMARY.md)). Tagging any published results by snapshot date is
recommended.

## What this is not

- Not a finding that any individual record is misclassified.
- Not a safety ranking of manufacturers, vehicles, ADS systems, or ADAS systems.
- Not a crash-rate or risk-rate model; SGO data does not include a reliable
  exposure denominator such as vehicle miles traveled at the needed granularity.
- Not an attempt to assign fault, crash causation, injury causation, or legal
  responsibility.
- Not affiliated with, endorsed by, or sponsored by NHTSA or any manufacturer.

## Responsible use

Because this project touches safety-critical public crash data:

- Flags are not findings. Every output means "inconsistent with reported label,
  recommended for human review," never "this record is wrong" or "this
  manufacturer misreported."
- No safety-rate claims. This project never states or implies that any
  automation system is safer or less safe than another.
- No re-identification attempts. Existing NHTSA redactions for PII, CBI, or
  other protected information must be respected as-is.
- Manufacturer-level aggregation is about classification patterns only. It must
  not be presented as a safety ranking.
- Phase 2 LLM inference is a heuristic signal. It is not a certified classifier
  and should be compared with Phase 1 and the reported label before review
  priority is assigned.

## Model limitations

- The public labels used for training and evaluation may themselves contain
  reporting errors.
- Structured fields may be incomplete, redacted, stale, or amended after a
  snapshot is downloaded.
- Phase 1 uses disagreement with the reported label as a review signal; without
  record-level ground truth, precision/recall calibration relies on synthetic
  label flips.
- Class balance and field availability differ between ADS and Level 2 ADAS
  records.
- Phase 2 depends on local Ollama model quality and prompt adherence. Narrative
  text can be sparse or redacted, and fallback mode is only a dry-run aid.
- Phase 3 named-investigation overlap is directional validation, not statistical
  proof.

## Outputs

Typical generated artifacts include:

- `data/processed/combined_audit_frame.csv`
- `data/processed/heuristic_flags.csv`
- `data/processed/entity_flag_rates.csv`
- `data/processed/classifier_flagged_records.csv`
- `data/processed/classifier_operating_point.json`
- `data/processed/llm_audit_results.csv`
- `data/processed/phase_comparison.csv`
- `data/processed/consensus_review_queue.csv`
- `data/processed/validation_report.json`
- `outputs/validation_note.md`
- `outputs/*.png`
- [results/](results/): small tracked excerpts + [RUN_SUMMARY.md](results/RUN_SUMMARY.md) for the canonical run
- [showcase/](showcase/): browseable UI over those tracked results

Raw CSVs/PDFs and generated pipeline outputs under `data/processed/` and
`outputs/` are intentionally ignored by git; `results/` and `showcase/` are
tracked for demos.

## Development process

Built AI-assisted under the staged process documented in
[AI-SDLC](https://github.com/nktarasios/AI-SDLC). See
[CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute.

## License

SGO-Audit is released under the MIT License. See [LICENSE](LICENSE).
