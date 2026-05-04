"""Option Evidence Scorer — inference-only predictor.

Hypothesis: For MCQ/select fields, explicitly presenting every option and asking
the LLM to score each one's supporting evidence from the source text will reduce
wrong_choice errors more than asking "what is the answer?" directly, because the
per-option evaluation forces the model to consider rejections explicitly
(e.g. "Not done" vs "Not assessed" vs "Not involved") rather than picking the
first plausible text.
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


MCQ_SCORING_PROMPT = """Evaluate each option against the source text.

Question: {title}
{detail}

{condition_note}
{guidance}

Options:
{options_block}

Source text:
{source_text}

For EACH option above, choose one:
- "SUPPORTED" — source text clearly states or strongly implies this
- "UNSUPPORTED" — source text does not support this (does not mention, or implies the opposite)

Return JSON: {{"option_text": "SUPPORTED", ...}}
If NO option is supported, return: {{"_verdict": "none"}}"""


TEXT_EXTRACTION_PROMPT = """Extract the exact answer from the source text.

Question: {title}
{guidance}
{condition_note}

Source text:
{source_text}

Return ONLY the exact value string including units as shown in source (e.g. "4.5 cm").
If the information is not present in the source, return "NONE".
Do not include quotes or JSON wrapper — return the raw answer string or "NONE"."""


def _build_mcq_prompt(field: dict[str, Any], source_text: str) -> str:
    title = field.get("title", field["name"])
    question = field.get("question", "")
    options = field.get("options", [])
    instructions = field.get("instructions", "")
    condition = field.get("condition")

    detail = f"Details: {question}" if question and question != title else ""

    condition_note = ""
    if condition:
        c = condition
        condition_note = (
            f"CONDITIONAL field. If {c['field']} does NOT {c['operator']} '{c['value']}', "
            "then NO option is supported (return _verdict: none)."
        )

    guidance = f"Guidance: {instructions}" if instructions else ""

    return MCQ_SCORING_PROMPT.format(
        title=title,
        detail=detail,
        condition_note=condition_note,
        guidance=guidance + "\n" if guidance else "",
        options_block="\n".join(f"  - {o}" for o in options),
        source_text=source_text,
    )


def _parse_mcq_response(field: dict[str, Any], response: str) -> Any:
    cleaned = response.strip()
    try:
        scores = json.loads(cleaned)
        if isinstance(scores, dict):
            if "_verdict" in scores:
                return None
            supported = [k.strip().strip('"').strip("'") for k, v in scores.items()
                         if isinstance(v, str) and v.strip().upper() == "SUPPORTED"]
            if field.get("multiple"):
                return supported if supported else None
            return supported[0] if supported else None
    except (json.JSONDecodeError, IndexError):
        pass
    return None


def _build_text_prompt(field: dict[str, Any], source_text: str) -> str:
    title = field.get("title", field["name"])
    instructions = field.get("instructions", "")
    condition = field.get("condition")

    guidance = f"Guidance: {instructions}" if instructions else ""
    condition_note = ""
    if condition:
        c = condition
        condition_note = f"CONDITIONAL: only if {c['field']} {c['operator']} '{c['value']}'."

    return TEXT_EXTRACTION_PROMPT.format(
        title=title,
        guidance=guidance,
        condition_note=condition_note,
        source_text=source_text,
    )


def _assemble_field(field: dict[str, Any], raw: Any) -> Any:
    if not field["is_mcq"]:
        return str(raw) if raw else ""

    if raw is None or raw == "":
        if field.get("multiple"):
            return {"answer": []}
        return {"answer": ""}

    if field.get("multiple"):
        if isinstance(raw, list):
            return {"answer": [str(x) for x in raw]}
        return {"answer": [str(raw)]}

    return {"answer": str(raw)}


def predict(input: str, call_llm) -> tuple[str, dict]:
    template_text = _extract_template(input)
    source_text = _extract_source(input)
    fields = _parse_fields(template_text)

    mcq_fields = [f for f in fields if f["is_mcq"]]
    text_fields = [f for f in fields if not f["is_mcq"]]

    output: dict[str, Any] = {}

    for f in mcq_fields:
        prompt = _build_mcq_prompt(f, source_text)
        response = call_llm(prompt)
        best = _parse_mcq_response(f, response)
        output[f["name"]] = _assemble_field(f, best)

    for f in text_fields:
        prompt = _build_text_prompt(f, source_text)
        response = call_llm(prompt)
        cleaned = response.strip().strip('"').strip("'")
        if cleaned.upper() == "NONE" or not cleaned:
            cleaned = ""
        output[f["name"]] = _assemble_field(f, cleaned)

    return output, {
        "strategy": "option_evidence_scorer",
        "mcq_fields": len(mcq_fields),
        "text_fields": len(text_fields),
        "llm_calls": len(fields),
    }