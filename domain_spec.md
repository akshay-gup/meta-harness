# Domain Spec: Transcript To React Hook Form State

## Domain Summary

The task is to fill a structured React Hook Form-compatible state from a text
transcript and an expanded form template. One evaluation unit is one case:
`transcript + form template -> filled form state`.

The form template contains field definitions, groups, nested groups, tables,
options, conditions, labels, field names, and other metadata. The expected output
is the same state shape the application already uses as React Hook Form output.
Intermediate extraction state may use any representation that improves model
performance, but final predictions must be assembled into the application state
shape.

The fixed solver provider is Together AI. The fixed solver model is
`deepseek-ai/DeepSeek-V4-Pro`. The Together setup is assumed to be covered for
PHI use by the user's confirmed HIPAA-appropriate account/contract settings.
The current benchmark config should use the LiteLLM provider-prefixed model
`together_ai/deepseek-ai/DeepSeek-V4-Pro`.

The proposer is OpenCode configured to use Together as well. PHI-bearing data may
be visible to proposer runs only under that HIPAA-covered Together setup. PHI
must still not be committed to git or copied into public/shareable artifacts.

Current data is limited to 5 real cases. This is a feasibility and baseline
implementation pass, not a statistically meaningful automated harness search.
The immediate goal is to verify that the harness interface, parsing, normalization,
evaluation, and logging work well enough to justify collecting more cases.

## Harness and Search Plan

Every candidate harness should satisfy the existing text-classification
`MemorySystem` interface:

- `predict(input: str) -> tuple[str, dict]`
- `learn_from_batch(batch_results: list[dict]) -> None`
- `get_state() -> str`
- `set_state(state: str) -> None`

For this domain, the `input` string represents one transcript plus one form
template and any evaluator-provided formatting around them. The formatted input
should include both normalized field metadata for extraction and the original
form-template JSON for component-specific details. The prediction should parse to
a completed form state.

The domain loader lives in `reference_examples/text_classification/form_filling_data.py`
to avoid colliding with the legacy `text_classification/data/` package used by
the older classification datasets.

The first baseline should be dynamic field-unit extraction:

- recursively traverse the template into fields, groups, nested groups, tables,
  repeated structures, and condition-controlled sections
- default to field-oriented extraction
- batch simple nearby scalar fields when they are short and low-risk, such as
  names, dates, IDs, yes/no fields, and simple text inputs
- split complex tables and repeated/nested structures into smaller extraction
  units when useful
- isolate ambiguous fields, fields with long option lists, and fields with
  complex conditions
- include relevant metadata per extraction unit: field name, label/title,
  question text, component/type, options, condition/dependency context, group
  path, and table/repeated structure information
- assemble intermediate extracted values into the exact React Hook Form state
  shape expected by the application

For MCQ/select fields, prompts should ask for exact canonical option strings.
The existing fuzzy matcher can normalize near matches, but weak matches should
remain unresolved or score incorrect rather than being forced to a bad option.

If the transcript does not support a field, the harness should not guess. It
should emit the RHF-compatible empty/default value for that field.

Changes in scope:

- prompt construction
- recursive schema traversal
- field grouping and dynamic extraction-unit planning
- field/table extraction strategy
- validation and repair calls
- parsing and output normalization
- final state assembly

Changes out of scope for the initial pass:

- changing the solver model during evaluation
- tuning on held-out test results
- committing PHI-bearing transcripts, targets, full prompts, or model outputs
- relying on hand-coded values from the 5 real cases

Initial baselines:

- current generic no-memory baseline
- schema-aware single-pass extraction
- strict per-field recursive extraction
- dynamic field-unit extraction
- generate-then-repair/normalize

Interface compliance should be tested by importing each candidate, instantiating
it with a stub LLM, running `predict()` on a synthetic transcript/template pair,
checking that the prediction parses as JSON or a JSON-like state, and verifying
that `get_state()`/`set_state()` round-trip.

## Evaluation Plan

The search/evaluation set currently contains 5 real transcript/template/target
cases. Use them only for validation/testing. With only 5 cases, scores are a
debug signal, not a reliable estimate of generalization.

Recommended temporary split:

- validation: 3 cases
- test: 2 cases

The split should remain fixed for a run. The final test set should not be used
to choose between candidate harnesses once more data exists. For the current
feasibility pass, test results may be inspected to debug the benchmark itself,
but any reported result should label the sample size and leakage risk clearly.

Primary metric:

- field-level accuracy after normalization

Scoring rules:

- conditional/non-applicable fields are ignored
- applicable fields are scored against the normalized target value
- missing or empty predictions are incorrect when the target has a value
- unsupported fields should be left empty rather than guessed
- full-form exact match is secondary, not primary

Normalization should be type-aware rather than truthy/falsy:

- strings: trim and normalize harmless whitespace/case differences where valid
- dates: normalize common date formats
- numeric text inputs: normalize formatting differences
- MCQ/select: map to canonical options with the existing fuzzy matcher
- multi-select: normalize order when order is not semantically meaningful
- valid negative/zero values such as `"No"`, `false`, `0`, and empty selections
  should not be treated as missing solely because of truthiness

Secondary metrics:

- parse success
- missing active fields
- extra active fields
- full-form exact match
- fuzzy match confidence/distance for MCQ/select fields
- model-call count
- token usage
- latency
- per-field and per-component error rates

Evaluation noise is unknown. With 5 examples, a single field or case can dominate
the aggregate score. The first goal is to inspect per-field diffs and recurring
failure modes.

One candidate evaluation may make multiple model calls per case. Cost is not a
current constraint, but latency and call count should still be logged.

Leakage risks:

- committing PHI-bearing data
- selecting candidates based on the 2-case test split
- hard-coding facts from the 5 real cases
- letting synthetic tests become too similar to real PHI cases

Mitigations:

- keep transcripts, targets, and full PHI-bearing logs out of git
- keep PHI-bearing `schemas/`, `targets/`, `transcripts/`, `logs/`, and
  `results/` paths ignored by git
- keep generated summaries free of PHI unless stored locally/private
- use synthetic toy forms for public tests and committed examples
- explicitly label the current run as feasibility-only
- expand the dataset before drawing conclusions from harness search

## Experience and Logging

Offline experience for the initial pass is limited. Real PHI cases may be used
by solver and proposer runs because both are intended to use the HIPAA-covered
Together setup, but they should remain local/private and uncommitted.

Useful future offline materials:

- synthetic transcript/template/target examples
- form-template structure notes
- known component semantics
- field normalizer documentation
- prior per-field error summaries without PHI
- examples of table/repeated-field structures

Online logs should preserve enough detail to debug failures:

- case id or manifest path
- template/schema path
- target path
- extraction units selected by the harness
- field paths and field metadata
- per-call prompt text
- raw model output
- parsed intermediate values
- assembled final RHF state
- normalized prediction
- normalized target
- per-field diff and score
- fuzzy match confidence/distance
- parse errors
- token counts
- latency and model-call count
- candidate source file and run metadata

PHI-bearing logs may include full prompts and outputs locally, but they must stay
in ignored/private paths and should not be committed. Public or proposer-readable
summary files should prefer aggregate metrics and redacted field-level error
categories unless intentionally running under the HIPAA-covered Together proposer.

Suggested directory structure:

- `reference_examples/text_classification/data/`: committed non-sensitive
  manifests and fixtures; PHI-bearing `schemas/`, `targets/`, and `transcripts/`
  stay ignored/private
- `reference_examples/text_classification/agents/`: baseline and generated
  harnesses
- `reference_examples/text_classification/logs/<run>/`: evaluation outputs,
  per-call logs, predictions, and diffs
- `reference_examples/text_classification/logs/<run>/opencode_sessions/`:
  proposer sessions when using OpenCode
- `reference_examples/text_classification/tests/fixtures/`: synthetic committed
  fixtures only

A small CLI for run history would be useful after the baseline works. Useful
commands would include:

- show run summary
- list candidates by validation field accuracy
- show worst fields
- show parse failures
- show per-case diff
- compare two candidates

## Open Questions and Unknowns

- Exact current React Hook Form state conventions for every component type:
  partially known from current data, but should be documented explicitly.
- Full list of supported form components: unknown.
- Existing fuzzy matcher thresholds and failure behavior: partially known;
  should be documented and tested.
- Date and numeric normalization rules: known conceptually, exact rules unknown.
- Best dynamic extraction-unit heuristics: unknown; start conservative and log
  decisions for inspection.
- Evaluation noise: unknown because there are only 5 cases.
- Long-term train/validation/test split: unknown until more cases are collected.
- Target candidate budget for future automated search: unknown. Default after
  data expansion should be 5-10 candidates for a first controlled search loop.
