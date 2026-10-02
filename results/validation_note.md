# Phase 3 validation note

Directional validation only: these ODI investigation IDs identify crashes NHTSA scrutinized for FSD reduced-visibility performance, not labeled ground truth for automation-level misclassification.

## Snapshot presence

- 7 of 13 reference SGO Report IDs appear in the current combined audit frame.
- Missing from this snapshot: 13781-11937, 13781-13211, 13781-13569, 13781-13633, 13781-13693, 13781-13788.

## Flag-set overlap

- heuristic_flags: 0 of 7 snapshot-present reference IDs; 0 of 13 total reference IDs. Status: loaded.
- classifier_flagged_records: 0 of 7 snapshot-present reference IDs; 0 of 13 total reference IDs. Status: loaded.
- phase1_phase2_agree_vs_reported: 0 of 7 snapshot-present reference IDs; 0 of 13 total reference IDs. Status: loaded.

## Caveats and responsible use

- This is a small validation sample; do not treat it as statistical proof.
- Flags are not findings. They identify records for human review only.
- The ODI resumes scrutinized these crashes for FSD reduced-visibility performance; they do not label the records as automation-level misclassifications.
- No output here supports manufacturer safety-rate or comparative safety claims.
