"""Option Contrast Reasoner — inference-only predictor.

Hypothesis: For hard MCQ fields, requiring the LLM to generate explicit accept/reject
reasoning with source quotes for EACH option BEFORE selecting the final answer will
reduce wrong_choice errors below the hard_field_verifier baseline, because structured
per-option reasoning forces critical evaluation of ALL alternatives rather than
defaulting to the first lexically convenient option.
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

        question_match = re.search(r"Question:\s*(.+)", block)
        if question_match:
            info["question"] = question_match.group(1).strip()

        fields.append(info)
    return fields


TEXT_FILL_PROMPT = """Extract the exact answers for these text fields from the source.

For each field return the exact value with units as shown in source (e.g. "4.5 cm").
If the information is not present in the source, return null.

Fields:
{fields_block}

Source text:
{source_text}

Return JSON: {{"field_name": "value_or_null", ...}}"""


CONTRAST_REASON_PROMPT = """Carefully evaluate every option for this question against the source text.

Question: {title}
{further}

{condition_note}
{instruction_block}

Options:
{options_block}

Source text:
{source_text}

Step 1: For EACH option above, answer:
  ACCEPT or REJECT — because [exact source text quote or note absence of evidence]

Step 2: Based on your analysis, state:
  FINAL_ANSWER: [the single best option text, or null if none fit]

CRITICAL RULES:
- "Not assessed" means NO mention of any evaluation for this item anywhere in source.
- "Not done" means the procedure is specifically mentioned as NOT performed.
- "Not involved" means evaluation was done and results are NORMAL/NEGATIVE.
- If the source says "normal", "NAD", "WNL", "unremarkable" → that means NOT involved, not unassessed.
- If source does not mention the procedure at all → Not done (for procedures) or Not assessed (for findings).
"""

OPTION_MEANING_GUIDE = (
    "\nOption disambiguation guide for this specific field:\n"
    "{guide}"
)

BM_BIOPSY_GUIDE = (
    "- \"Yes\" — source states BM biopsy SHOWS involvement\n"
    "- \"No\" — source states BM biopsy performed and NEGATIVE\n"
    "- \"Not done\" — no mention of BM biopsy being performed"
)

CNS_GUIDE = (
    "- \"Imaging abnormality\" — CNS imaging shows abnormality, CSF not mentioned\n"
    "- \"CSF Positive\" — CSF analysis positive for lymphoma\n"
    "- \"Both\" — imaging abnormality AND CSF positive\n"
    "- \"Not assessed\" — NO CNS evaluation mentioned at all (no imaging, no CSF)\n"
    "- \"Not involved\" — evaluated and NORMAL (e.g. 'CNS WNL', 'CNS NAD', 'no CNS involvement')"
)

FIELD_GUIDES = {
    "mcq_1755701329003": BM_BIOPSY_GUIDE,
    "mcq_1755701655242": CNS_GUIDE,
}


def _assemble_field(field: dict[str, Any], value: Any) -> Any:
    if value is None or value == "":
        if field["is_mcq"]:
            return {"answer": [] if field.get("multiple") else ""}
        return ""

    if not field["is_mcq"]:
        return str(value)

    if field.get("multiple"):
        items = value if isinstance(value, list) else [value]
        return {"answer": [str(x) for x in items]}
    return {"answer": str(value)}


def _parse_contrast_response(response: str) -> str | None:
    match = re.search(r"FINAL_ANSWER\s*:\s*(.+)", response, re.IGNORECASE)
    if match:
        val = match.group(1).strip()
        if val.lower() == "null":
            return None
        return val.strip().strip('"').strip("'")

    match = re.search(r'"final_answer"\s*:\s*"([^"]*)"', response)
    if match:
        val = match.group(1)
        if val.lower() == "null":
            return None
        return val

    match = re.search(r'(?:answer|selection)\s*:\s*"?([^"\n]+)"?', response, re.IGNORECASE)
    if match:
        val = match.group(1).strip().strip('"').strip("'")
        if val.lower() == "null":
            return None
        return val

    return None


def predict(input: str, call_llm) -> tuple[str, dict]:
    source_text = _extract_source(input)
    template_text = _extract_template(input)
    fields = _parse_fields(template_text)

    text_fields = [f for f in fields if not f["is_mcq"]]
    mcq_fields = [f for f in fields if f["is_mcq"]]

    output: dict[str, Any] = {}

    # Phase 1: Fill all text fields in one batch call
    if text_fields:
        text_desc = []
        for f in text_fields:
            desc = f"{f['name']} ({f.get('title', '')})"
            if f.get("instructions"):
                desc += f" — Note: {f['instructions']}"
            if f.get("condition"):
                c = f["condition"]
                desc += f" [CONDITIONAL: only if {c['field']} {c['operator']} '{c['value']}']"
            text_desc.append(desc)

        text_prompt = TEXT_FILL_PROMPT.format(
            fields_block="\n".join(text_desc),
            source_text=source_text,
        )
        text_response = call_llm(text_prompt)

        text_values = {}
        try:
            data = json.loads(text_response)
            if isinstance(data, dict):
                text_values = data
        except json.JSONDecodeError:
            tv_raw = extract_json_field(text_response, "")
            if tv_raw:
                try:
                    text_values = json.loads(tv_raw)
                except json.JSONDecodeError:
                    pass

        for f in text_fields:
            raw = text_values.get(f["name"])
            if isinstance(raw, str) and raw.lower() == "null":
                raw = None
            output[f["name"]] = _assemble_field(f, raw)
    else:
        for f in text_fields:
            output[f["name"]] = _assemble_field(f, None)

    # Phase 2: For each MCQ field, use contrast reasoning
    for f in mcq_fields:
        title = f.get("title", f["name"])
        question = f.get("question", "")
        options = f.get("options", [])
        instructions = f.get("instructions", "")
        condition = f.get("condition")
        multiple = f.get("multiple")

        further = f"Details: {question}" if question and question != title else ""

        condition_note = ""
        if condition:
            c = condition
            condition_note = (
                f"\nCONDITIONAL FIELD: only fill if {c['field']} {c['operator']} '{c['value']}'."
            )

        instruction_block = f"Guidance: {instructions}" if instructions else ""

        guide = FIELD_GUIDES.get(f["name"], "")
        option_guide_block = OPTION_MEANING_GUIDE.format(guide=guide) if guide else ""

        prompt = CONTRAST_REASON_PROMPT.format(
            title=title,
            further=further,
            condition_note=condition_note,
            instruction_block=instruction_block + option_guide_block,
            options_block="\n".join(f"  - {o}" for o in options),
            source_text=source_text,
        )

        if multiple:
            prompt += "\n\nNote: Multiple answers allowed. In FINAL_ANSWER, list all accepted options as a JSON array."

        response = call_llm(prompt)
        parsed = _parse_contrast_response(response)

        if multiple and parsed:
            try:
                arr = json.loads(parsed)
                if isinstance(arr, list):
                    parsed_list = arr
                else:
                    parsed_list = [parsed]
            except json.JSONDecodeError:
                parsed_list = [parsed]
            output[f["name"]] = _assemble_field(f, parsed_list)
        else:
            output[f["name"]] = _assemble_field(f, parsed)

    return output, {
        "strategy": "option_contrast_reasoner",
        "num_mcq_fields": len(mcq_fields),
        "num_text_fields": len(text_fields),
    }