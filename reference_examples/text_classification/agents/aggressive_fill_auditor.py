"""Aggressive-Fill Evidence Auditor — inference-only predictor.

Hypothesis: Reversing the plan-then-fill approach to fill-then-audit will reduce
empty_prediction errors that plagued the condition_planner. An aggressive first
pass fills every field (never skipping), then a second audit pass checks each
field for truly-supported source evidence and empties fields without evidence.
This avoids the planner over-pruning problem because the fill pass includes
everything, and the audit pass is a simpler binary check per field.
"""

import json
import re
from typing import Any

from ..memory_system import extract_json_field


def _extract_template(input_text: str) -> str:
    idx = input_text.find("Input form template for validation:")
    if idx == -1:
        return ""
    text = input_text[idx:]
    if "Expected output state shape:" in text:
        text = text.split("Expected output state shape:")[0]
    return text


def _extract_source(input_text: str) -> str:
    if "Source text:\n" in input_text:
        return input_text.split("Source text:\n", 1)[1].strip()
    return ""


def _parse_fields(template_text: str) -> list[dict[str, Any]]:
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

        info["is_mcq"] = "Type: mcq" in block or "Type: select" in block
        info["multiple"] = "Multiple answers allowed" in block

        opt_match = re.search(r"Options:\s*(.+)", block)
        if opt_match:
            info["options"] = [o.strip() for o in opt_match.group(1).split(";")]

        cond_match = re.search(
            r"Only fill when.+\((\S+)\)\s+(\S+)\s+(.+)", block
        )
        if cond_match:
            info["condition"] = {
                "field": cond_match.group(1),
                "operator": cond_match.group(2),
                "value": cond_match.group(3).strip(),
            }

        inst_match = re.search(r"Instructions:\s*(.+)", block)
        if inst_match:
            info["instructions"] = inst_match.group(1).strip()

        fields.append(info)
    return fields


AGGRESSIVE_FILL_PROMPT = """Fill ALL fields below using the source text.

DO NOT skip any field — provide your best answer for every single field.
For MCQ/select fields choose the exact option text from the provided list.
For text fields extract the exact value with units as shown in source (e.g. "4.5 cm").
If you are unsure, still provide a best guess — do not leave any field blank.

Fields:
{fields_desc}

Source text:
{source_text}

Return a JSON object: {{"field_name": "answer_string", ...}}"""


EVIDENCE_AUDIT_PROMPT = """For each field below, determine whether the current answer has sufficient evidence.

A field should be marked "no_evidence" if:
- The source text does NOT contain information about this field
- A conditional field does NOT have its parent condition met
- The required procedure/test was mentioned as NOT performed
- The field references a specific measurement/metric that does not appear in the source

Current filled answers:
{current_answers}

Source text:
{source_text}

Return a JSON object mapping field_name → true or false, where true means "the source text contains evidence for this field" and false means "this field should be empty because no source evidence supports it"."""


def _parse_json_response(response: str) -> dict:
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    raw = extract_json_field(response, "")
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {}


def _assemble_field(field: dict[str, Any], raw: Any, has_evidence: bool) -> Any:
    is_mcq = field["is_mcq"]
    multiple = field.get("multiple", False)

    if not has_evidence or raw is None or raw == "":
        if is_mcq:
            return {"answer": [] if multiple else ""}
        return ""

    if not is_mcq:
        return str(raw)

    if multiple:
        items = raw if isinstance(raw, list) else [str(raw)]
        return {"answer": [str(i) for i in items]}
    return {"answer": str(raw)}


def predict(input: str, call_llm) -> tuple[str, dict]:
    template_text = _extract_template(input)
    source_text = _extract_source(input)
    fields = _parse_fields(template_text)

    field_desc_lines = []
    for f in fields:
        line = f"{f['name']} ({f.get('title', '')})"
        if f.get("options"):
            line += f" — Options: [{', '.join(f['options'])}]"
        if f.get("condition"):
            c = f["condition"]
            line += f" [CONDITIONAL: only if {c['field']} {c['operator']} '{c['value']}']"
        field_desc_lines.append(line)

    fill_prompt = AGGRESSIVE_FILL_PROMPT.format(
        fields_desc="\n".join(field_desc_lines),
        source_text=source_text,
    )
    fill_response = call_llm(fill_prompt)
    filled = _parse_json_response(fill_response)

    audit_prompt = EVIDENCE_AUDIT_PROMPT.format(
        current_answers=json.dumps(filled, indent=2, ensure_ascii=False),
        source_text=source_text,
    )
    audit_response = call_llm(audit_prompt)
    evidence_map = _parse_json_response(audit_response)

    output: dict[str, Any] = {}
    for f in fields:
        name = f["name"]
        raw = filled.get(name)
        has_ev = evidence_map.get(name, True)
        if not isinstance(has_ev, bool):
            has_ev = True
        output[name] = _assemble_field(f, raw, has_ev)

    return output, {
        "strategy": "aggressive_fill_auditor",
        "num_fields": len(fields),
        "evidenced": sum(1 for v in evidence_map.values() if v is True),
        "un_evidenced": sum(1 for v in evidence_map.values() if v is False),
    }