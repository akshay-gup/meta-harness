"""Per-Field Decomposer — inference-only predictor.

Hypothesis: Isolating each field into a separate mini-LLM query with the full
source text will reduce wrong_choice and wrong_text_or_number errors compared
to a single-pass fill, because the LLM can focus attention on one question at
a time without distraction from other fields or output formatting concerns.
Deterministic code handles wrapper assembly to eliminate type errors.
"""

import json
import re
from typing import Any


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

        question_match = re.search(r"Question:\s*(.+)", block)
        if question_match:
            info["question"] = question_match.group(1).strip()

        fields.append(info)
    return fields


def _build_field_prompt(field: dict[str, Any], source_text: str) -> str:
    title = field.get("title", field["name"])
    question = field.get("question", "")
    options = field.get("options", [])
    multiple = field.get("multiple", False)
    instructions = field.get("instructions", "")
    condition = field.get("condition")

    lines = [f"Question: {title}"]
    if question and question != title:
        lines.append(f"Details: {question}")

    if options:
        lines.append(f"Allowed options: [{', '.join(options)}]")
        if multiple:
            lines.append("Multiple selections allowed — return a JSON array of option texts.")

    if condition:
        c = condition
        lines.append(
            f"NOTE: This field is CONDITIONAL — only fill if {c['field']} {c['operator']} '{c['value']}'."
        )
        lines.append("If the condition is NOT met or unclear, return null.")

    if instructions:
        lines.append(f"Guidance: {instructions}")

    prompt = "\n".join(lines)
    prompt += f"\n\nSource text:\n{source_text}\n\n"

    if options:
        prompt += "Return ONLY the exact option text string (or JSON array if multiple). If none fit, return null."
    else:
        prompt += "Return ONLY the exact value string including units as shown in source (e.g. \"4.5 cm\"). If not found, return null."

    return prompt


def _assemble_value(field: dict[str, Any], raw: Any) -> Any:
    if isinstance(raw, str):
        raw = raw.strip().strip('"').strip("'")

    is_empty = raw is None or raw == "" or (isinstance(raw, str) and raw.lower() == "null")

    if not field["is_mcq"]:
        return str(raw) if not is_empty else ""

    if is_empty:
        if field.get("multiple"):
            return {"answer": []}
        return {"answer": ""}

    if field.get("multiple"):
        if isinstance(raw, list):
            return {"answer": [str(x) for x in raw]}
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, list):
                return {"answer": [str(x) for x in parsed]}
        except (json.JSONDecodeError, TypeError):
            pass
        return {"answer": [str(raw)]}

    return {"answer": str(raw)}


def predict(input: str, call_llm) -> tuple[str, dict]:
    template_text = _extract_template(input)
    source_text = _extract_source(input)
    fields = _parse_fields(template_text)

    output: dict[str, Any] = {}
    for f in fields:
        prompt = _build_field_prompt(f, source_text)
        response = call_llm(prompt)
        output[f["name"]] = _assemble_value(f, response)

    return output, {
        "strategy": "per_field_decomposer",
        "num_fields": len(fields),
        "llm_calls": len(fields),
    }