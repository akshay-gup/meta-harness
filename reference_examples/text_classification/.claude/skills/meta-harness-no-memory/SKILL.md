---
name: meta-harness-no-memory
description: Run one iteration of inference-only Meta-Harness evolution. Use when training examples are scarce, disabled, or the goal is to avoid memory tricks and instead build no-memory/decomposition predictors.
---

# Meta-Harness (Inference-Only Evolution)

Run ONE iteration of evolution. Do all work in the main session. Do NOT delegate.

This variant is for **no-memory or memory-light systems**. Candidates do not need
to implement the full `MemorySystem` class. They may expose a plain inference-only
API, and the harness will wrap it. Prefer schema parsing, decomposition,
verification, and exact JSON assembly over retrieval or example memorization.

**You do NOT run benchmarks.** The outer loop runs validation after you write
`pending_eval.json`.

## Critical Constraints

- Implement exactly 3 new systems every iteration.
- At least 2 of the 3 candidates must use a plain inference-only API rather than
  subclassing `MemorySystem`.
- Candidates must work with zero training examples.
- Do not create candidates whose main mechanism is example retrieval, few-shot memory,
  contrastive memory, error memory, or stored target values.
- Do not edit `config.yaml` just to register candidates. The benchmark auto-discovers
  files in `agents/`.
- Do not hardcode validation rows, field answers, patient/source text, dataset names,
  run paths, or literal target values from logs.
- Use only sanitized diagnostics for failure patterns. Treat raw validation labels as
  off-limits even if files are readable.

## What Good Candidates Look Like

Build systems that reason longer at inference time:

- **Schema parser + deterministic assembler**: parse field IDs, types, options,
  conditions, and output skeleton; always return exact expected JSON shape.
- **Applicability planner**: decide which fields have source evidence before filling.
- **Targeted field verifier**: after a draft answer, re-check hard/ambiguous fields
  for source evidence and option semantics.
- **Domain-generic MCQ disambiguator**: distinguish states like unknown/not assessed,
  negative/not involved, not done, and blank/not applicable by reading field wording
  and source evidence.
- **Selective decomposition**: one full-form pass plus extra LLM calls only for fields
  with high sanitized error counts.

Bad candidates merely tune constants, retrieve more examples, or store more memory.

## Workflow

### Step 0: Reports

Check the reports directory from the task prompt. If a past iteration has entries in
`evolution_summary.jsonl` but no short report, write one in `reports/`.
Keep each report <=30 lines.

### Step 1: Analyze

Read:

- `evolution_summary.jsonl`
- `frontier_val.json`
- `config.yaml`
- `reports/field_diagnostics.md`
- per-agent `val_diagnostics.json` only for sanitized field/error counts
- top agent source files

Formulate 3 falsifiable hypotheses. At least 2 must target inference-time behavior,
not training-time memory.

### Step 2: Prototype

Prototype before final implementation.

For each candidate, write a small `/tmp/` script that exercises the core logic:

- schema/field parser
- condition parser
- JSON assembler
- draft/verify prompt construction
- hard-field selector

Use synthetic snippets or sanitized patterns. Do not copy validation targets into
prototype code. Delete `/tmp/` scripts before finishing.

### Step 3: Implement

For each candidate:

1. Create `agents/<name>.py`.
2. Prefer one of the plain inference-only APIs below.
3. Keep prediction usable with no prior training or state.
4. Prefer deterministic parsing and assembly around LLM calls.
5. Validate through the same loader the benchmark uses:

```bash
PYTHONPATH=.. uv run python -c "from text_classification.inner_loop import load_memory_system; m = load_memory_system('agents/<name>.py', lambda prompt: '{}'); print('OK', type(m).__name__)"
```

Use `extract_json_field(response, "final_answer") or response` for final answer
extraction when parsing an LLM response.

## Candidate Patterns To Try

Use these as starting points, but still implement three distinct mechanisms:

- **Planner then fill**: one call creates an applicability plan, one call fills only
  planned fields, deterministic code restores missing empty fields.
- **Draft then verifier**: one call drafts the whole form, second call reviews only
  sanitized hard fields and returns patch JSON.
- **Hard-field decomposer**: fill the full form once, then make small field-specific
  calls for hard fields named by diagnostics.
- **Option semantics checker**: deterministic prompt section explaining how to choose
  between negative, not done, not assessed, not applicable, and blank.
- **Strict assembler**: parse the output skeleton and coerce every answer into the
  expected field ID and MCQ/text shape.

## Accepted Agent APIs

Plain function:

```python
def predict(input: str, call_llm) -> tuple[str, dict]:
    response = call_llm(prompt)
    return completed_form_json, {"strategy": "name"}
```

Plain class:

```python
class Agent:
    def __init__(self, call_llm):
        self.call_llm = call_llm

    def predict(self, input: str) -> tuple[str, dict]:
        response = self.call_llm(prompt)
        return completed_form_json, {"strategy": "name"}
```

Factory:

```python
def build_agent(call_llm):
    return Agent(call_llm)
```

The harness also accepts legacy `MemorySystem` subclasses, but do not use them
unless there is a strong reason.

Return a completed form state, not template metadata. If you parse LLM JSON,
you may import `extract_json_field` from `..memory_system`.

## Output

Write the exact `pending_eval.json` path from the task prompt:

```json
{
  "iteration": 1,
  "candidates": [
    {
      "name": "candidate_name",
      "file": "agents/candidate_name.py",
      "hypothesis": "falsifiable claim",
      "axis": "exploitation|exploration",
      "base_system": "what it builds on",
      "components": ["inference_only", "schema_parser", "verifier"]
    }
  ]
}
```

Then output:

```text
CANDIDATES: name1, name2, name3
```
