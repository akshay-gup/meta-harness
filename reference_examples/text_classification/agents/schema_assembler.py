"""Schema-Parsing Deterministic Assembler — inference-only predictor.

Hypothesis: Parsing the form schema from the input to extract field types, options,
and conditions, using deterministic output assembly to enforce exact per-field wrapper
format (MCQ→{"answer": str}, Input→plain str, multichoice→{"answer": [str]}), will
eliminate format/structure errors and type mismatches that penalize raw LLM output.
"""

import json
import re
from typing import Any


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


def _parse_field_info(template_text: str) -> list[dict[str, Any]]:
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

        question_match = re.search(r"Question:\s*(.+)", block)
        if question_match:
            info["question"] = question_match.group(1).strip()

        fields.append(info)
    return fields


def _parse_skeleton(skeleton_text: str) -> dict[str, Any]:
    # The skeleton is a JSON object; extract the first complete JSON object
    idx = skeleton_text.find("{")
    if idx == -1:
        return {}
    depth = 0
    for i, c in enumerate(skeleton_text[idx:], idx):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(skeleton_text[idx : i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def _build_value_extraction_prompt(fields: list[dict[str, Any]], source_text: str) -> str:
    """Build a prompt asking LLM to extract values in structured format, no wrapping."""
    field_desc = []
    for i, f in enumerate(fields, 1):
        parts = [f"{i}. {f['name']}"]
        if f.get("title"):
            parts[0] += f" ({f['title']})"
        if f.get("question") and f["question"] != f.get("title", ""):
            parts.append(f"   Question: {f['question']}")
        if f.get("options"):
            parts.append(f"   Allowed: [{', '.join(f['options'])}]")
        if f.get("multiple"):
            parts.append("   Multiple selections: list values in an array")
        if f.get("instructions"):
            parts.append(f"   Note: {f['instructions']}")
        if f.get("condition"):
            cond = f["condition"]
            parts.append(
                f"   CONDITIONAL: Only fill if {cond['field']} {cond['operator']} '{cond['value']}'"
            )
        field_desc.append("\n".join(parts))

    fields_block = "\n\n".join(field_desc)

    prompt = f"""Extract field values from the source text.

For EACH field determine:
  - "value": the exact answer from source text
  - "empty": true if the field should NOT be filled (no source evidence, or condition not met, or data not available)

For MCQ/select fields return the option text. For text fields return the exact text or number with units as shown in source.

Return ONLY a JSON object mapping field_name → value string or array of strings. For empty/unfilled fields use null.

Fields:
{fields_block}

Source text:
{source_text}

Answer in the format: {{"field_name": "value", ...}}"""
    return prompt


def _assemble_output(field_name: str, value: Any, field_info: dict[str, Any]) -> Any:
    """Build the exact harness-expected output format for a field."""
    if not field_info.get("is_mcq"):
        if field_info.get("multiple"):
            if isinstance(value, list):
                return str(value) if len(value) == 1 else value
            return str(value) if value else ""
        return str(value) if value else ""

    if field_info.get("multiple"):
        items = value if isinstance(value, list) else ([value] if value else [])
        return {"answer": items}
    if value is None or value == "":
        return {"answer": ""}
    return {"answer": str(value)}


def _build_output(
    extracted_values: dict[str, Any], fields: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build the completed form dict in exact harness format."""
    fields_by_name = {f["name"]: f for f in fields}
    output: dict[str, Any] = {}

    # Output all fields in schema order
    for f in fields:
        name = f["name"]
        raw = extracted_values.get(name)
        output[name] = _assemble_output(name, raw, f)

    return output


def predict(input: str, call_llm) -> tuple[str, dict]:
    template_text = _extract_block(input, "Input form template for validation:", "Expected output state shape:")
    source_text = input.split("Source text:\n", 1)[1] if "Source text:\n" in input else ""

    fields = _parse_field_info(template_text)

    extraction_prompt = _build_value_extraction_prompt(fields, source_text)
    raw_response = call_llm(extraction_prompt)

    extracted = {}
    try:
        data = json.loads(raw_response)
        if isinstance(data, dict):
            extracted = data
    except json.JSONDecodeError:
        try:
            from ..memory_system import extract_json_field
            values_raw = extract_json_field(raw_response, "")
            if values_raw:
                extracted = json.loads(values_raw)
        except (json.JSONDecodeError, Exception):
            extracted = {}

    output = _build_output(extracted, fields)
    return output, {
        "strategy": "schema_assembler",
        "num_fields": len(fields),
    }