"""Condition-Aware Applicability Planner — inference-only predictor.

Hypothesis: Pre-parsing field conditions to create an applicability plan — filling
only fields whose parent conditions are satisfied or that have direct source evidence,
and leaving inactive/conditional fields explicitly empty — will reduce
predicted_when_target_empty errors and prevent hallucination on conditional fields.
"""

import json
import re
from typing import Any

from ..memory_system import extract_json_field


def _extract_block(text: str, start_marker: str, end_marker: str | None = None) -> str:
    start = text.find(start_marker)
    if start == -1:
        return ""
    content = text[start:]
    if end_marker:
        end = content.find(end_marker)
        if end != -1:
            content = content[:end]
    return content


def _extract_source_text(input_text: str) -> str:
    if "Source text:\n" in input_text:
        return input_text.split("Source text:\n", 1)[1].strip()
    return ""


def _parse_field_specs(template_text: str) -> list[dict[str, Any]]:
    blocks = re.split(r"\n(?=\d+\.\s)", template_text.strip())
    fields = []
    for block in blocks:
        name_match = re.search(r"\d+\.\s+(\S+)\s*[—\-–]", block)
        if not name_match:
            continue
        name = name_match.group(1)
        info: dict[str, Any] = {"name": name}

        title_match = re.search(r"\d+\.\s+\S+\s*[—\-–]\s*(.+)", block)
        if title_match:
            info["title"] = title_match.group(1).split("\n")[0].strip()

        if "Type: mcq" in block or "Type: select" in block:
            info["is_mcq"] = True
        else:
            info["is_mcq"] = False

        if "Multiple answers allowed" in block:
            info["multiple"] = True
        else:
            info["multiple"] = False

        opt_match = re.search(r"Options:\s*(.+)", block)
        if opt_match:
            info["options"] = [o.strip() for o in opt_match.group(1).split(";")]

        cond_match = re.search(r"Only fill when.+\((\S+)\)\s+(\S+)\s+(.+)", block)
        if cond_match:
            info["condition"] = {
                "field": cond_match.group(1),
                "operator": cond_match.group(2),
                "value": cond_match.group(3).strip(),
            }

        inst_match = re.search(r"Instructions:\s*(.+)", block)
        if inst_match:
            info["instructions"] = inst_match.group(1).strip()

        question_match = re.search(r"Question:\s*(.+)", block)
        if question_match:
            info["question"] = question_match.group(1).strip()

        fields.append(info)
    return fields


PLANNING_PROMPT = """Determine which fields are applicable given the source text.

For each field, decide:
- "fill": true if there is source evidence to fill this field
- "fill": false if no source evidence exists, OR the field cannot be filled (e.g. the field is conditional and its parent condition is not met)

A conditional field (marked 'Only fill when ...') should ONLY be filled if:
  1. Its parent condition IS met (the parent field's answer matches the condition value)
  2. There IS evidence in the source text for this field

Return a JSON object mapping every field_name → true or false.

Fields:
{fields_block}

Source text:
{source_text}

Answer: {{"field_name": true, ...}}"""


FILL_PROMPT = (
    "Fill ONLY the fields listed below from the source text.\n\n"
    "For each field you must fill, provide the exact answer.\n"
    "- For MCQ/select: return the exact option text\n"
    "- For text: return the exact value with units as shown in source\n\n"
    "Fields to fill:\n"
    "{fields_to_fill}\n\n"
    "Fields that should remain EMPTY (do NOT fill these — they will be left blank):\n"
    "{fields_to_skip}\n\n"
    "Source text:\n"
    "{source_text}\n\n"
    "Return a JSON object mapping field_name → value for ONLY the fields you fill.\n"
    'Answer: {{"field_name": "value", ...}}'
)


def _assemble_full_output(
    filled: dict[str, Any], fields: list[dict[str, Any]]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for f in fields:
        name = f["name"]
        raw = filled.get(name)
        if raw is None:
            if f.get("is_mcq"):
                output[name] = {"answer": [] if f.get("multiple") else ""}
            else:
                output[name] = ""
        else:
            if f.get("is_mcq"):
                if f.get("multiple"):
                    items = raw if isinstance(raw, list) else ([raw] if raw else [])
                    output[name] = {"answer": items}
                else:
                    output[name] = {"answer": str(raw)}
            else:
                output[name] = str(raw)
    return output


def predict(input: str, call_llm) -> tuple[str, dict]:
    source_text = _extract_source_text(input)

    template_text = _extract_block(
        input,
        "Input form template for validation:",
        "Expected output state shape:",
    )
    fields = _parse_field_specs(template_text)

    # Phase 1: Plan applicability
    field_names = [f["name"] for f in fields]
    fields_desc = []
    for f in fields:
        desc = f"{f['name']} ({f.get('title', '')})"
        if f.get("options"):
            desc += f" — Options: [{', '.join(f['options'])}]"
        if f.get("condition"):
            c = f["condition"]
            desc += f" [CONDITIONAL: only if {c['field']} {c['operator']} '{c['value']}']"
        fields_desc.append(desc)

    plan_prompt = PLANNING_PROMPT.format(
        fields_block="\n".join(fields_desc),
        source_text=source_text,
    )

    plan_response = call_llm(plan_prompt)
    plan_raw = extract_json_field(plan_response, "")
    if plan_raw:
        try:
            plan = json.loads(plan_raw)
        except json.JSONDecodeError:
            plan = {}
    else:
        try:
            plan = json.loads(plan_response)
        except json.JSONDecodeError:
            plan = {}
    if not isinstance(plan, dict):
        plan = {}

    to_fill = [name for name in field_names if plan.get(name, True)]
    to_skip = [name for name in field_names if name not in to_fill]

    # Phase 2: Fill only applicable fields
    if not to_fill:
        output = _assemble_full_output({}, fields)
        return output, {
            "strategy": "condition_planner",
            "filled_count": 0,
            "skipped_count": len(to_skip),
        }

    fill_desc_lines = []
    for f in fields:
        if f["name"] not in to_fill:
            continue
        desc = f"{f['name']} ({f.get('title', '')})"
        if f.get("options"):
            desc += f" — Options: [{', '.join(f['options'])}]"
        if f.get("instructions"):
            desc += f" — Note: {f['instructions']}"
        fill_desc_lines.append(desc)

    fill_prompt = FILL_PROMPT.format(
        fields_to_fill="\n".join(fill_desc_lines),
        fields_to_skip=", ".join(to_skip) if to_skip else "(none)",
        source_text=source_text,
    )

    fill_response = call_llm(fill_prompt)
    fill_raw = extract_json_field(fill_response, "")
    if fill_raw:
        try:
            filled = json.loads(fill_raw)
        except json.JSONDecodeError:
            filled = {}
    else:
        try:
            filled = json.loads(fill_response)
        except json.JSONDecodeError:
            filled = {}
    if not isinstance(filled, dict):
        filled = {}

    output = _assemble_full_output(filled, fields)
    return output, {
        "strategy": "condition_planner",
        "filled_count": len(to_fill),
        "skipped_count": len(to_skip),
    }