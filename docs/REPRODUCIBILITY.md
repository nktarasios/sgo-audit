# Reproducibility

SGO-Audit is designed to be re-run by strangers without asking the author.

## What “reproducible” means here

1. **Same code, same Archive snapshot** → same tracked metrics in `results/`.
2. **Same code, newer NHTSA CSVs** → fresh flags tagged with a new `data_vintage`.
3. **Same method, another structured crash dataset** → map columns, keep the
   three-signal pattern (rules → model → narrative reader → consensus queue).

## Re-run on NHTSA data

```bash
pip install -r requirements.txt
python scripts/download_data.py                 # Archive-2021-2025 (canonical)
python scripts/run_all.py --llm-backend transformers --llm-sample 50
python -m pytest
```

Current third-amended CSVs:

```bash
python scripts/download_data.py --current --overwrite
python scripts/run_all.py --current --llm-backend transformers --llm-sample 50
```

Always read outputs against `data_vintage` in `results/ingest_metadata.json`
(also surfaced in `results/RUN_SUMMARY.md`).

## Adapt to another dataset

The pipeline assumes two label classes (ADS vs Level 2 ADAS) and a table of
structured fields plus an optional narrative.

| Piece | Where to change |
|---|---|
| Column names / aliases | `src/fields.py` |
| Hard consistency rules | `src/heuristics.py` |
| Model features / threshold | `src/classifier.py` |
| Narrative prompt / parse | `src/llm_auditor.py` |
| Fusion ranking | `src/consensus.py` |
| Download paths | `scripts/download_data.py` |

Keep the product contract:

- outputs are **review flags**, not findings
- no safety rankings / crash rates
- every published run carries an explicit data vintage

## Offline tests (no download)

```bash
python -m pytest
```

Tests use tiny fixtures under `tests/fixtures/` only.

## Showcase UI

```bash
python scripts/build_showcase_data.py
python scripts/serve_showcase.py
```
