# Product Requirements Document
## Project: SGO-Audit, Detecting Automation-Level Misclassification in NHTSA Crash Reports

**Owner:** Nabil Khoury
**Status:** Ready for implementation
**License intent:** MIT (open source)
**Doc type:** PRD, pairs with `TECHNICAL_IMPLEMENTATION.md`

---

## 1. Summary

NHTSA's Standing General Order (SGO) 2021-01 requires manufacturers and operators to report crashes involving Level 2 ADAS and ADS (higher automation) vehicles. NHTSA has publicly acknowledged that reporting entities sometimes misclassify which automation level was actually engaged in a given incident, and that NHTSA is working with reporting entities to correct this over time. Separately, safety advocates, Consumer Reports, and members of Congress have raised concerns that the dataset lacks exposure normalization (no vehicle-miles-traveled denominator), making raw crash counts easy to misread or misrepresent.

**SGO-Audit** is an open-source tool that flags SGO incident records whose reported automation-level classification appears inconsistent with the rest of the record, treating classification-consistency as a detectable signal, the same way a monitoring system detects sensor false positives/negatives. It does not attempt to adjudicate ground truth on its own; it produces ranked, explainable flags for human/regulatory review, and validates its approach where possible against NHTSA's own published investigation findings.

---

## 2. Problem Statement

**Primary problem:** A safety-critical, self-reported federal dataset, used by NHTSA, journalists, researchers, and litigants to evaluate real-world AV/ADAS safety, has a known, acknowledged classification-quality issue (ADS vs. Level 2 ADAS mislabeling) with no existing public tool to systematically detect it at scale. Today this appears to be caught case-by-case, through manual investigation.

**Secondary problem (context, not solved directly):** The dataset has no exposure denominator, so raw counts cannot support rate-based safety claims. This project does not attempt to fix that; it explicitly documents it as a known limitation of the underlying data so that no output from this project is misread as a rate-based safety judgment.

**Why this matters beyond a portfolio exercise:** Improving classification-consistency confidence in this dataset has direct value to NHTSA's own stated data-quality goals, to journalists and researchers who currently take the labels at face value, and to any future contributor who wants to build safety analytics on top of this data.

---

## 3. Goals & Non-Goals

### Goals
- Build an explainable, reproducible pipeline that flags likely automation-level misclassifications in public SGO data.
- Validate the approach against real NHTSA investigation findings where record-level ground truth is publicly available (see Section 5, Phase 3).
- Release as clean, documented, open-source code that a researcher, journalist, or engineer could rerun on a future SGO data refresh.
- Produce a clear writeup demonstrating detection-threshold/tradeoff reasoning applied to a real, current, substantive problem.

### Non-goals
- This project does **not** claim to determine ground truth for any individual crash record. All outputs are "flagged for review," never "confirmed misclassified."
- This project does **not** attempt to normalize crash counts by exposure/VMT, that data isn't reliably available at the needed granularity, and pretending otherwise would be a factually unsupportable claim.
- This project does **not** make comparative safety claims about any named manufacturer's overall safety record. Any manufacturer-level aggregation is about **reporting/classification pattern**, not vehicle safety performance.
- No production deployment, no live data pipeline, no API, this is a research/audit tool, run on demand against downloaded data snapshots.

---

## 4. Users & Use Cases

| User | Use case |
|---|---|
| Me (primary) | A worked example of detection/threshold reasoning on real data |
| Open-source contributors | Researchers/engineers who want to extend the classifier, add data sources, or audit new SGO releases |
| Journalists / safety researchers | A tool to sanity-check classification consistency before citing SGO numbers in reporting |
| NHTSA-adjacent policy community | Illustrative example of the kind of independent data-quality tooling this dataset invites |

---

## 5. Scope, Phased

### Phase 0, Scaffolding (build first, cheap, foundational)
- Data ingestion + cleaning pipeline for the public ADAS and ADS SGO CSVs.
- Rule-based heuristic flags (e.g., stated automation level inconsistent with recorded pre-crash movement pattern).
- Reporting-entity trend tracking: flag rate per manufacturer over time, framed as a data-quality KPI, not a safety KPI.

### Phase 1, Classical ML classifier (primary deliverable)
- Train a classical model (gradient boosting or logistic regression baseline) to predict automation level (ADS vs. Level 2 ADAS) from structured fields.
- Flag records where the model's high-confidence prediction disagrees with the reported label.
- Evaluate using precision/recall and a threshold-selection methodology consistent with prior detection-threshold work (explicit operating-point tradeoff, not just raw accuracy).

### Phase 2, Local LLM narrative auditor (GPU-accelerated, RTX 5080)
- Run a local open-weight LLM (via Ollama) against each incident's narrative text field to infer the described automation behavior independently of the structured label.
- Compare LLM-inferred classification against both the reported label and the Phase 1 model's prediction.
- The interesting output is not "the LLM flags things" but **where Phase 1 and Phase 2 disagree**: those are the highest-value review candidates, since two independent methods failing to agree signals real ambiguity.
- Runs fully locally, no data leaves the machine, no API cost. Worth stating explicitly given this is safety-sensitive federal data.

### Phase 3, Ground-truth validation against known NHTSA investigations
- NHTSA's own Engineering Analysis documents cite specific SGO report ID numbers it has already scrutinized in named investigations.
- Cross-reference flagged records against these known, already-investigated report IDs as a real (if small-sample) validation set, this is the strongest evidence the project's flags mean something, since it's checked against NHTSA's own findings rather than synthetic labels.

### Explicitly out of scope for all phases
- No predictive safety-outcome modeling (crash severity prediction): that was the prior draft's scope and is intentionally dropped in favor of the classification-consistency problem.
- No dashboard/web app in v1 (may be a post-v1 community contribution).
- No exposure-normalization attempt (Section 3, non-goals).

---

## 6. Success Criteria

1. Phase 0+1 run end-to-end on a fresh download of the public CSVs, producing a ranked, explainable flag list with precision/recall reasoning documented.
2. Phase 2 runs fully locally on the RTX 5080 without requiring any paid API, and produces a clear agreement/disagreement comparison against Phase 1.
3. Phase 3 successfully cross-references at least the known investigation report IDs found in the NHTSA ODI documents referenced in this PRD, and states plainly whether the project's flags overlap with NHTSA's own scrutinized records.
4. The repository is clean enough that an outside contributor could clone it, follow the README, and reproduce every phase without needing to ask you anything.
5. A one-page writeup exists that a non-technical reader (e.g., a journalist) could read in under 5 minutes and understand exactly what was found and why it's real.

---

## 7. Responsible Use, Disclaimers & Ethics

Because this touches a safety-critical, federally-collected dataset about real crashes, the following must be stated explicitly in the repo (README and any published writeup), not left implicit:

- **Flags are not findings.** Every output is "inconsistent with reported label, recommended for human review", never "this record is wrong" or "this manufacturer misreported."
- **No safety-rate claims.** This project never states or implies that any automation system is safer or less safe than another. It is exclusively about reporting/classification consistency.
- **No re-identification attempts.** All existing NHTSA redactions (PII, CBI) must be respected as-is; no attempt to infer redacted content.
- **Manufacturer-level aggregation is about classification patterns only**, clearly labeled as such wherever it appears, to avoid the output being screenshotted out of context as a safety ranking.
- **Model limitations section required** in the README: sample size caveats, the fact that structured-field labels themselves may be incomplete, and that LLM-based inference (Phase 2) is a heuristic signal, not a certified classifier.

---

## 8. Open Source Plan

- **License:** MIT, permissive, standard for a research/tooling project, encourages reuse by researchers and journalists.
- **README must include:** problem statement (plain-language version of Section 2), quickstart, data-source attribution and link to NHTSA's SGO page, responsible-use section (Section 7), and a "what this is not" section.
- **CONTRIBUTING.md:** lightweight, how to add a new data refresh, how to propose a new heuristic/feature, code style (black/ruff for Python).
- **Versioning:** tag releases against the SGO data snapshot date used (NHTSA amends/expands the CSVs over time), so results are always reproducible against a stated data vintage.
- **Citation/attribution:** clear statement that this is an independent project, not affiliated with or endorsed by NHTSA or any manufacturer.

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Output misread as "proof" a manufacturer misreported | Section 7 disclaimers, explicit "flagged for review" language everywhere in code output and docs |
| Small validation sample in Phase 3 | State sample size plainly; frame as directional validation, not statistical proof |
| Structured-field data quality issues masking real signal | Document known data limitations (redactions, revision history) in README, don't silently paper over them |
| Scope creep into a full crash-severity/safety-rate project | This PRD's Section 5 "out of scope" is binding, any change proposing scope beyond it should be redirected back to this doc |
| LLM inference cost/time on large record counts | Phase 2 should run on a sampled subset first to validate approach before running on the full dataset |

---

## 10. Timeline (effort-sized, not calendar-locked)

- Phase 0: 1 weekend
- Phase 1: 1 weekend
- Phase 2: 1 weekend (GPU work, but Ollama setup is fast)
- Phase 3: a few focused hours, mostly manual cross-referencing against a small number of known report IDs
- Writeup + open-source polish: a few focused hours

Total: roughly 3–4 weekends, matching your original 1.5–2 year runway with plenty of margin, this does not need to be rushed.

---

## 11. References (for the README's attribution section)

- NHTSA Standing General Order on Crash Reporting: https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting
- NHTSA SGO 2021-01 Data Element Definitions (data dictionary): available via the same page
- NHTSA ODI Engineering Analysis documents (for Phase 3 ground-truth report IDs): static.nhtsa.gov/odi/inv/, specific investigation PDFs should be sourced fresh at build time
- Consumer Reports comments to NHTSA on SGO reporting (context for Section 7's exposure-normalization caveat)
