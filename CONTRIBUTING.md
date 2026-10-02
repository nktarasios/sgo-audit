# Contributing

SGO-Audit is serious research tooling for public SGO data-quality review. Keep
changes narrow, reproducible, and explicit about uncertainty.

## Responsible-use baseline

- Outputs are review flags, not findings of misclassification.
- Do not add safety rankings, safety-rate claims, fault analysis, or crash
  causation claims.
- Respect NHTSA redactions as-is. Do not infer or reconstruct redacted content.
- Manufacturer/entity aggregations must be labeled as classification-consistency
  patterns only.

## Adding a heuristic rule

1. Add one named function in `src/heuristics.py`.
2. Use the signature already established by existing rules:

   ```python
   def rule_name(record: Mapping[str, Any]) -> tuple[bool, str]:
       ...
   ```

3. Return `(True, "short human-readable reason")` only when the record should be
   flagged for review. Return `(False, "")` otherwise.
4. Add the rule to the `RULES` tuple.
5. Add focused tests in `tests/test_heuristics.py` using synthetic records.
6. Keep the reason string factual and non-accusatory. Prefer "field X is
   inconsistent with field Y" over any wording that implies confirmed reporting
   error.

## Proposing a data refresh

NHTSA amends and republishes SGO files over time. To propose a refresh:

1. Run `python scripts/download_data.py --current --overwrite` in a local working
   copy.
2. Record the resulting `DATA_VINTAGE.txt` contents and source URLs.
3. Re-run Phase 0 and Phase 1 commands from the README.
4. Summarize row counts, flag counts, and any schema changes.
5. Do not commit raw NHTSA CSVs, PDFs, downloaded investigation files, or large
   generated outputs. Commit only code, tests, docs, and small fixtures.

If the refresh changes column names or semantics, update `src/fields.py` and add
tests that demonstrate the new schema behavior.

## Phase 2 and Phase 3 contributions

- Phase 2 real narrative auditing requires local Ollama. The keyword fallback is
  only for dry-run/smoke-test checks and should not be described as LLM output.
- Phase 3 validation should remain a small curated cross-reference against named
  SGO report IDs in public NHTSA investigation materials, such as EA26002 and
  PE24031. Treat overlap as directional, not statistical proof.

## Code style

- Format Python with `black`.
- Lint with `ruff`.
- Keep module-level docstrings that explain each phase's role.
- Prefer small pure functions with explicit tests.
- Avoid unrelated refactors in data-processing changes.

Suggested local checks:

```bash
black .
ruff check .
python -m pytest
```

## Tests

- Use synthetic fixtures under `tests/fixtures/`.
- Do not require full raw NHTSA downloads for unit tests.
- Cover ingestion redaction handling, heuristic rule behavior, classifier
  threshold/flag behavior, LLM parsing, and comparison categories.
- When adding CLI behavior, include at least one test for the command-facing
  helper function so future docs stay aligned with implementation.
