# Technical Implementation Doc
## Project: SGO-Audit
**Pairs with:** `PRD.md`: read that first for problem statement, scope, and non-goals.
**Audience:** implementation reference + human review.

---

## 1. Architecture Overview

```
sgo-audit/
├── data/
│   ├── raw/                     # untouched downloaded CSVs + data dictionary PDF
│   └── processed/               # cleaned/merged intermediate files
├── src/
│   ├── ingest.py                 # Phase 0: load + clean raw CSVs
│   ├── heuristics.py              # Phase 0: rule-based consistency flags
│   ├── entity_trends.py           # Phase 0: per-manufacturer flag-rate tracking
│   ├── classifier.py              # Phase 1: classical ML model
│   ├── llm_auditor.py             # Phase 2: local LLM narrative auditor (Ollama)
│   ├── compare.py                 # Phase 2: Phase1 vs Phase2 disagreement analysis
│   └── validate.py                # Phase 3: ground-truth cross-reference
├── notebooks/
│   ├── 01_explore.ipynb
│   ├── 02_classifier_eval.ipynb
│   └── 03_llm_comparison.ipynb
├── tests/
│   └── test_ingest.py, test_heuristics.py, test_classifier.py, ...
├── README.md
├── CONTRIBUTING.md
├── LICENSE                        # MIT
├── requirements.txt
└── pyproject.toml (optional, for packaging as a CLI later)
```

Design principle: **each phase is a standalone, independently runnable module.** A contributor should be able to run Phase 0 alone and get value (the heuristic flags), without needing Phase 1 or 2. This matters for open-source usability as much as for your own build discipline.

---

## 2. Tech Stack

- **Python 3.11+**
- **pandas**: ingestion/cleaning
- **scikit-learn** (baseline) and optionally **xgboost/lightgbm**: Phase 1 classifier
- **Ollama**: local LLM serving for Phase 2, running on the RTX 5080
- **sentence-transformers** (optional, only if narrative clustering is added later, not required for the core three phases)
- **matplotlib**: the handful of charts (precision/recall curve, entity trend line, agreement/disagreement breakdown)
- **pytest**: tests
- **jupyter**: exploratory notebooks (not the primary deliverable code path, logic should live in `src/`, notebooks call into it)

No web framework, no database, no cloud deployment. This is a local, reproducible research tool.

---

## 3. Phase 0, Data Ingestion, Cleaning, Heuristics, Entity Trends

### 3.1 Ingestion (`ingest.py`)
- Load both the ADAS (Level 2) and ADS public CSVs separately, do not merge on load; keep the `reported_automation_level` field distinct per source file, since that field itself is the thing being audited.
- Load the data dictionary PDF's field definitions as a reference (manually transcribe the key fields into a small config/constants file, don't attempt PDF parsing for this, it's a one-time, low-volume task better done by hand).
- Handle `[REDACTED, MAY CONTAIN CONFIDENTIAL BUSINESS INFORMATION]` and similar redaction strings as explicit `None`/NaN, tagged with a `was_redacted` boolean column per field, this preserves the signal that "we don't know" is different from "this field was empty."
- Log and preserve the file's revision/vintage date (NHTSA periodically amends the CSVs): store this as a run-level metadata field so every output can be traced to a specific data snapshot.

### 3.2 Heuristic flags (`heuristics.py`)
- Encode a small set (start with 3–5, don't over-build) of hand-written consistency rules, e.g.:
  - Stated automation level is ADS, but pre-crash movement / narrative language pattern matches typical Level-2-supervised behavior (e.g., explicit mention of driver takeover, hands-on-wheel language).
  - Stated automation level is Level 2 ADAS, but the crash circumstances match ADS-only operational patterns (e.g., no driver present, geofenced robotaxi context).
- Each rule should be a separate, named, independently testable function returning a boolean flag + a short reason string, this reason string is what makes the output explainable, which matters for the open-source audience ("here's why the model flagged this one").
- Do not try to be exhaustive here, 3–5 well-reasoned rules beat 15 noisy ones. This module is scaffolding for Phase 1's features, not the final answer.

### 3.3 Entity trend tracking (`entity_trends.py`)
- Aggregate heuristic + (later) model flag rates by reporting entity and by month/quarter.
- Output one simple trend chart: flag rate over time, per entity, explicitly labeled "classification-consistency flag rate," never "safety incident rate", see PRD Section 7 on why that labeling distinction matters.

---

## 4. Phase 1, Classical ML Classifier

### 4.1 Approach
- **Task framing:** binary/multiclass classification predicting `reported_automation_level` (ADS vs. Level 2 ADAS) from structured fields, deliberately *excluding* the label itself and any field that trivially encodes it.
- **Features:** crash circumstances, roadway type, speed band, pre-crash movement, whether a driver was present/engaged (if available in the data), heuristic flag outputs from Phase 0 (these become model features, not just standalone outputs).
- **Model:** start with logistic regression as an interpretable baseline; add gradient boosting (xgboost/lightgbm) as a stronger second model. Keep both, the comparison between a fully-interpretable model and a stronger black-box model is itself worth reporting.
- **The actual "flag":** a record is flagged when the model's predicted label disagrees with the reported label at a chosen confidence threshold.

### 4.2 Threshold methodology (this is the load-bearing analytical piece)
- Do not just report accuracy. Sweep the confidence threshold and produce a precision/recall curve for "flag this record as likely misclassified."
- Choose and justify one recommended operating point explicitly (e.g., "threshold chosen to flag the top X% most-confident disagreements, keeping expected false-flag rate below Y%"): write it up as a plain threshold decision.
- Document class imbalance (ADS vs. Level 2 ADAS crash counts are very likely imbalanced) and how it's handled (class weighting, stratified evaluation): this is a real methodological point, not boilerplate.

### 4.3 Evaluation artifacts
- One precision/recall (or ROC) curve chart.
- One feature-importance chart (for the interpretable model): this is what makes the output explainable rather than a black box, which matters for open-source trust.
- A ranked CSV of flagged records with their reasons (heuristic rule triggered + model disagreement confidence).

---

## 5. Phase 2, Local LLM Narrative Auditor (RTX 5080)

### 5.1 Setup
- Install Ollama; pull an open-weight instruction model sized comfortably for a 16GB-class GPU (RTX 5080): a ~8B-parameter model (e.g., Llama 3.1 8B instruct or similar current open-weight option) run quantized (Q4/Q5) will run fast locally and leave headroom. Confirm current best-available open-weight options at build time since this space moves quickly, don't hardcode an assumption about which model is best without checking what's current.
- No API keys, no cloud calls, this runs entirely on your machine, which is worth stating explicitly in the README as a deliberate design choice (no cost, no data leaving the machine for safety-sensitive federal data).

### 5.2 Prompting approach
- For each incident's narrative text (where not fully redacted), prompt the model with a constrained task: "Based on this crash narrative, was the vehicle most likely operating as a fully automated driving system, or as a Level 2 driver-assist system requiring an attentive human driver? Answer with one label and a one-sentence justification citing specific phrases from the narrative."
- Keep the prompt narrow and single-purpose, do not ask the model to also assess severity, fault, or safety implications. Scope discipline applies here as much as anywhere else in this doc.
- Run on a sample first (50–100 records) to validate the prompt produces sensible, consistent output before running the full dataset.

### 5.3 Comparison (`compare.py`)
- Three-way comparison per record: reported label, Phase 1 model prediction, Phase 2 LLM inference.
- The headline output is the **disagreement matrix**: cases where all three agree (low priority for review), cases where Phase 1 and Phase 2 agree with each other but differ from the reported label (highest-priority flags, two independent methods converging), and cases where Phase 1 and Phase 2 disagree with each other (genuinely ambiguous, interesting to discuss, not a strong flag).
- This three-way framing is the most defensible, most interesting output of the whole project, it's the part worth leading with in any writeup.

---

## 6. Phase 3, Ground-Truth Validation

- From NHTSA's published Engineering Analysis / investigation documents, extract the specific SGO report ID numbers already named as scrutinized in a real investigation (these appear directly in NHTSA's own investigation PDFs, listed by ID).
- Check whether any of those specific report IDs appear in your Phase 1/2 flagged set.
- Report this as a small, explicitly-labeled validation note: "N of the M records already named in NHTSA's own [investigation name] appear in our flagged set", state the sample size plainly; do not inflate a small overlap into a strong claim (see PRD Section 9, risks).
- This is manual, targeted work, a handful of report IDs, not a bulk automated matching pipeline. Don't over-engineer this step.

---

## 7. Testing Strategy

- Unit tests for each heuristic rule in isolation (given a synthetic record, does the rule fire as expected).
- Unit tests for the ingestion/cleaning logic, especially redaction handling.
- A small fixture dataset (a handful of hand-constructed rows, not the real data) for fast CI-style testing without needing the full CSVs present.
- No requirement for exhaustive coverage, test the logic that would be embarrassing to get wrong (redaction handling, threshold calculation), not every line.

---

## 8. Open Source Packaging Checklist

- [ ] `LICENSE`: MIT
- [ ] `README.md`: problem statement (plain language), quickstart (`pip install -r requirements.txt`, how to download the raw data, how to run each phase), responsible-use section (pulled from PRD Section 7), data-vintage/versioning note, attribution/non-affiliation statement
- [ ] `CONTRIBUTING.md`: how to add a heuristic rule, how to propose a new data refresh, code style
- [ ] `requirements.txt` pinned to working versions
- [ ] Each `src/` module has a docstring explaining its role and linking back to the relevant PRD section
- [ ] No secrets, no API keys anywhere (Phase 2 is fully local, nothing to leak)
- [ ] Repo description and topics tagged for discoverability (e.g., `nhtsa`, `av-safety`, `data-quality`, `open-source`)

---

## 9. Build Sequence (in order)

1. Scaffold the repo structure from Section 1.
2. Implement Phase 0 (`ingest.py`, `heuristics.py`, `entity_trends.py`) fully, with tests, before touching Phase 1.
3. Implement Phase 1 (`classifier.py`) using Phase 0's cleaned output + heuristic flags as features. Produce the precision/recall curve and justify one operating point.
4. Implement Phase 2 (`llm_auditor.py`, `compare.py`) only after Phase 1 is working end-to-end. Validate the Ollama setup on a small sample before running full-scale.
5. Implement Phase 3 (`validate.py`) as a small, mostly-manual cross-reference script.
6. Write `README.md`, `CONTRIBUTING.md`, finalize `requirements.txt`.
7. At every step: if a proposed addition isn't explicitly listed in this doc or in `PRD.md`'s scope, stop and confirm before building it, the biggest risk to this project is scope creep past the phased plan.

---

## 10. Open Items to Confirm Before/During Build

1. Current best-available open-weight instruct model for Ollama on a 16GB-class GPU, check at build time, this doc intentionally doesn't hardcode a choice since the landscape moves fast.
2. Exact current column names/schema in the live SGO CSVs (verify against the freshly-downloaded data dictionary, not just this doc's Section 3.1 description).
3. Which specific NHTSA investigation documents to pull report IDs from for Phase 3, start with the ones already referenced in your research (e.g., recent FSD-related Engineering Analysis documents) and expand only if time allows.
