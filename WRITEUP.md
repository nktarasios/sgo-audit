# SGO-Audit one-page summary

## What was checked

NHTSA's public Standing General Order crash-reporting data separates records
reported as Level 2 ADAS from records reported as ADS. SGO-Audit checks whether
those reported automation-level labels look internally consistent with the rest
of each public record.

This is a data-quality audit, not a safety study.

## What the end-to-end run found

On NHTSA's **Archive-2021-2025** public CSVs (about **4,589** latest-version
records after cleaning: 1,899 ADS + 2,690 ADAS):

- Phase 0 heuristic rules produced **37** classification-consistency flags.
- Phase 1 structured modeling produced **20** high-confidence model-disagreement
  flags at the chosen operating point (LightGBM, threshold **0.59**,
  calibration `none`). Synthetic-flip precision / recall at that point:
  **0.912 / 0.378**. The strongest structured signal is operational context
  (especially Driver / Operator Type), so high label-prediction accuracy is
  expected; the useful output is the small disagreement set, not overall
  accuracy.
- Phase 2 ran a real local LLM (`Qwen/Qwen2.5-1.5B-Instruct` via transformers)
  on **50** non-redacted narratives. In that sample: **41** three-way
  agreements, **9** Phase1/Phase2 disagreements, and **0** cases where both
  methods agreed against the reported label.
- The Phase 2.5 consensus queue fused those signals into **66** review
  candidates (all currently `single_signal` tier on this run).
- Phase 3 found **7 of 13** named PE24031/EA26002 SGO Report IDs in the Archive
  snapshot and **0** overlap with the project's current flag sets. That is a
  small-sample directional check only, those ODI IDs are reduced-visibility
  investigation records, not misclassification ground truth.

Tracked machine-readable outputs live in `results/`. A browseable showcase UI
is in `showcase/` (`python scripts/serve_showcase.py`).

## Why flags are not findings

A flag means "this record deserves human review because one signal disagrees
with another." It does not mean the reported label is wrong, that a manufacturer
misreported, or that any automation system is safer or less safe than another.

The underlying public data can be revised, redacted, sparse, or incomplete.
There is also no reliable exposure denominator in this dataset, so raw crash
counts cannot be converted into safety rates.

The useful output is a short, explainable review queue: records where the label,
structured fields, model prediction, or narrative signal do not line up cleanly.

## Independence

SGO-Audit is independent and is not affiliated with or endorsed by NHTSA or any
manufacturer.
