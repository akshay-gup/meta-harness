"""Evidence Span Extractor — inference-only predictor.

Hypothesis: Extracting relevant source-text spans for each field BEFORE answering
will reduce wrong_choice and wrong_text_or_number errors because the extraction
phase forces the LLM to locate and quote specific source evidence for every answer,
preventing the model from hallucinating values or reasoning from general medical
knowledge rather than the provided text.
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


EVIDENCE_EXTRACTION_PROMPT = """For each field below, extract the EXACT span of text from the source that provides evidence.

For each field:
- "span": quote the exact source text that answers this field
- "empty": true if NO source text provides evidence for this field, or if the field is conditional and its parent condition is not met

For MCQ/select fields, only extract the sentence(s) describing the relevant finding, procedure, or result.
For text fields, extract the exact measurement or description from source.

Fields:
{fields_block}

Source text:
{source_text}

Return JSON: {{"field_name": {{"span": "exact quote", "empty": true/false}}, ...}}"""


ANSWER_FROM_EVIDENCE_PROMPT = """Using ONLY the extracted evidence text below, determine the answer for this field.

Field: {title}
{further}

{option_block}
{instruction_block}

Extracted evidence:
"{evidence}"

{condition_note}

Return ONLY the exact answer string. For MCQ/select, return the option text. For text, return the value with units. If the field should remain empty, return null."""


def _build_evidence_prompt(fields: list[dict[str, Any]], source_text: str) -> str:
    field_desc = []
    for f in fields:
        parts = [f"{f['name']} ({f.get('title', '')})"]
        if f.get("question") and f.get("question") != f.get("title", ""):
            parts.append(f"   Question: {f['question']}")
        if f.get("options"):
            parts.append(f"   Options: [{', '.join(f['options'])}]")
        if f.get("condition"):
            c = f["condition"]
            parts.append(f"   CONDITIONAL: only if {c['field']} {c['operator']} '{c['value']}'")
        field_desc.append("\n".join(parts))
    return EVIDENCE_EXTRACTION_PROMPT.format(
        fields_block="\n\n".join(field_desc),
        source_text=source_text,
    )


def _assemble_field(field: dict[str, Any], value: Any) -> Any:
    is_mcq = field["is_mcq"]
    multiple = field.get("multiple", False)

    if value is None or value == "":
        if is_mcq:
            return {"answer": [] if multiple else ""}
        return ""

    if not is_mcq:
        return str(value)

    if multiple:
        items = value if isinstance(value, list) else [value]
        return {"answer": [str(x) for x in items]}
    return {"answer": str(value)}


def _build_single_answer_prompt(field: dict[str, Any], evidence: str) -> str:
    title = field.get("title", field["name"])
    question = field.get("question", "")
    options = field.get("options", [])
    instructions = field.get("instructions", "")
    condition = field.get("condition")

    further = f"Details: {question}" if question and question != title else ""
    option_block = f"Allowed options:\n" + "\n".join(f"  - {o}" for o in options) if options else ""
    instruction_block = f"Guidance: {instructions}" if instructions else ""

    condition_note = ""
    if condition:
        c = condition
        condition_note = (
            f"\nThis is a CONDITIONAL field. If {c['field']} does NOT {c['operator']} "
            f"'{c['value']}', return null even if evidence mentions it."
        )

    return ANSWER_FROM_EVIDENCE_PROMPT.format(
        title=title,
        further=further,
        option_block=option_block,
        instruction_block=instruction_block,
        evidence=evidence if evidence else "(none — no relevant text found in source)",
        condition_note=condition_note,
    )


def predict(input: str, call_llm) -> tuple[str, dict]:
    source_text = _extract_source(input)
    template_text = _extract_block(
        input,
        "Input form template for validation:",
        "Expected output state shape:",
    )
    fields = _parse_fields(template_text)

    evidence_prompt = _build_evidence_prompt(fields, source_text)
    evidence_response = call_llm(evidence_prompt)

    evidence_map: dict[str, dict[str, Any]] = {}
    try:
        data = json.loads(evidence_response)
        if isinstance(data, dict):
            evidence_map = data
    except json.JSONDecodeError:
        ev_raw = extract_json_field(evidence_response, "")
        if ev_raw:
            try:
                evidence_map = json.loads(ev_raw)
            except json.JSONDecodeError:
                pass

    output: dict[str, Any] = {}
    for f in fields:
        name = f["name"]
        ev_entry = evidence_map.get(name, {})
        if isinstance(ev_entry, dict):
            span = ev_entry.get("span", "")
            is_empty = ev_entry.get("empty", False)
        else:
            span = str(ev_entry) if ev_entry else ""
            is_empty = False

        if is_empty or not span:
            output[name] = _assemble_field(f, None)
            continue

        answer_prompt = _build_single_answer_prompt(f, span)
        answer_response = call_llm(answer_prompt)
        cleaned = answer_response.strip().strip('"').strip("'")

        if cleaned.lower() == "null" or not cleaned:
            output[name] = _assemble_field(f, None)
        else:
            output[name] = _assemble_field(f, cleaned)

    return output, {
        "strategy": "evidence_span_extractor",
        "num_fields": len(fields),
        "evidence_fields": sum(
            1 for e in evidence_map.values()
            if isinstance(e, dict) and e.get("span") and not e.get("empty")
        ),
    }