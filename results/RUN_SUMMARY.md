# SGO-Audit end-to-end run summary

Generated (UTC): `2026-07-23T23:40:02.362776+00:00`
Data vintage: **Archive-2021-2025**
LLM backend: **transformers**
LLM sample size: **50**
Ollama model requested: `Qwen/Qwen2.5-1.5B-Instruct`

## Canonical data vintage (tracked results)

**Canonical snapshot for tracked `results/` and this summary: `Archive-2021-2025`.**

| | prior known-good (`aebae38`) | this tracked run (`710669c`+) |
|---|---|---|
| CLI / download | `scripts/run_all.py --current` | `scripts/run_all.py` (archive default) |
| `data_vintage` label | `current-third-amended` | `Archive-2021-2025` |
| Source URL | `static.nhtsa.gov/.../sgo-2021-01/` (live third-amended) | `.../sgo-2021-01/Archive-2021-2025/` |
| Raw columns | 116 | 137 |
| ADS / ADAS cleaned | 1200 / 1702 | 1899 / 2690 |
| Operating threshold | 0.91 | 0.59 |
| Synth P / R | 0.972 / 0.955 | 0.912 / 0.378 |
| Calibration | n/a (equiv. `none`) | `none` (explicit default) |

Run-level vintage is written by ingest into `ingest_metadata.json` /
`results/ingest_metadata.json` as `data_vintage` (TECHNICAL_IMPLEMENTATION §3.1)
and surfaced on the `Data vintage:` line above by `scripts/run_all.py`
(`resolve_data_vintage` prefers the ingest field over the raw
`DATA_VINTAGE.txt` line).

`--current` remains supported for live third-amended refreshes, but is **not**
the baseline for tracked metrics going forward.

### What drove the P/R / threshold shift (do not conflate)

- **Snapshot change, essentially all of it.** Same precision-first rule
  (≥80% synthetic precision, then max recall), same model family (`lightgbm`),
  same calibration (`none`). Re-fitting on the larger Archive corpus
  (~+58% cleaned rows; archive-only optional fields such as `ADS Equipped?`)
  re-selects the operating point at 0.91 → 0.59, with synth P/R
  0.972/0.955 → 0.912/0.378. Precision stays near the ≥80% floor; recall drops
  because the Archive synthetic-flip surface is harder at that floor.
- **Calibration decision, none of the shift.** Default stayed `none`. The
  sigmoid/isotonic table in the consolidate writeup was an Archive-only side
  study and was not applied to either compared run.
- **Other, not Phase-1 P/R drivers.** Consensus queue (new) and Phase 2
  sample size (still 50) do not change classifier threshold/P/R. Heuristic
  flag count 4 → 37 tracks the larger Archive corpus under the same rules.

## Branch / merge state

- Integration base going forward: `main` (PR #4 merged).
- Merged: PR #2 (`AGENTS.md` setup docs) + PR #3 (consensus queue +
  optional `--calibration`).
- Out of scope for this pass (next, not now): narrative-text Phase 1 features,
  Phase 2 strengthening, Phase 3 ground-truth expansion.

## Phase 0, ingest + heuristics
- ADS cleaned rows: `1899`
- ADAS cleaned rows: `2690`
- Combined (approx): `4589`
- Heuristic-flagged records: `37`

## Phase 1, classical classifier
- Selected flag model: `lightgbm`
- Probability calibration: `none` (default `none`; see decision below)
- Operating threshold: `0.590`
- Synthetic-flip precision / recall: `0.912` / `0.378`
- Model disagreement flags: `20`
- Rationale: Selected the threshold that meets the synthetic mislabel precision target of 80%, maximizing recall within that precision constraint.

### Precision-first threshold + calibration default

- Threshold selection is precision-first in `src/classifier.py`
  (`choose_operating_point`): require synthetic precision ≥ 80%, then
  maximize recall within that constraint.
- `--calibration` default reviewed against Archive-2021-2025:
  `none` ≈ `isotonic` precision (~0.91) with fewer, higher-confidence
  flags; `sigmoid`/`isotonic` expanded the queue at lower mean
  confidence. **Default remains `none`** (opt-in exploration only).

## Phase 2, narrative auditor + disagreement matrix
- Backend used: `transformers`
- Disagreement counts: `{"all_agree": 41, "only_one_method_available": 0, "phase1_phase2_agree_vs_reported": 0, "phase1_phase2_disagree": 9}`

## Consensus review queue (Phase 0/1/2 fusion)
- Records in queue: `66`
- Tier counts: `{"single_signal": 66, "three_signal_consensus": 0, "two_signal_consensus": 0}`
- Records with Phase 2 evidence: `9`

## Phase 3, NHTSA investigation ID cross-reference
- Reference IDs: `13`
- Present in snapshot: `7`
- Missing from snapshot: `13781-11937, 13781-13211, 13781-13569, 13781-13633, 13781-13693, 13781-13788`
- Flag overlaps: `{"heuristic_flags": 0, "classifier_flagged_records": 0, "phase1_phase2_agree_vs_reported": 0}`

## Responsible use

Flags are review signals only, not confirmed misclassifications, not
manufacturer safety rankings, and not exposure-normalized crash rates.

## Artifacts in this folder

- `ingest_metadata.json`
- `classifier_operating_point.json`
- `classifier_metrics.json`
- `disagreement_matrix.json`
- `validation_report.json`
- `heuristic_flags.csv`
- `classifier_flagged_records.csv`
- `consensus_review_queue.csv`
- `consensus_summary.json`
- `validation_note.md`
