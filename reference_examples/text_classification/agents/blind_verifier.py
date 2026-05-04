"""Blind Verifier — inference-only predictor.

Hypothesis: A second-pass verifier that reviews hard MCQ fields WITHOUT showing the
Phase 1 answer for the field under review will outperform the hard_field_verifier's
approach of displaying the current filled form (which includes the target field's
answer), because showing the pre-existing answer creates anchoring bias that prevents
independent re-evaluation of source evidence against the available options.
"""

import json
import re
from typing import Any

from ..memory_system import extract_json_field


FILL_PROMPT = (
    "Fill the form from the source text.\n\n"
    "{input}\n\n"
    "Return the completed form state in `final_answer`. Use the output state shape "
    "provided in the input when present. Do not return the blank input template or "
    "template metadata such as component/title/config/options.\n\n"
    "**Answer in this exact JSON format:**\n"
    '{{"reasoning": "[brief reasoning]", "final_answer": {{"completed": "form state"}}}}\n'
)

BLIND_REVIEW_PROMPT = (
    "You are independently answering a single field question from a medical form. "
    "Answer based ONLY on the source text below. Do NOT guess or use external knowledge.\n\n"
    "Field: **{field_title}**\n\n"
    "Allowed options and their meanings:\n"
    "{option_meanings}\n\n"
    "Guidance:\n"
    "{instructions}\n\n"
    "Source text:\n"
    "{source_text}\n\n"
    "What is the correct answer for this field? "
    "Return ONLY the exact option text string. If the field should be empty, return null."
)

BM_BIOPSY_MEANINGS = (
    "- \"Yes\" — the source explicitly states that bone marrow biopsy SHOWS lymphoma involvement.\n"
    "- \"No\" — the source states bone marrow biopsy was performed and is NEGATIVE (no involvement found).\n"
    "- \"Not done\" — no bone marrow biopsy is mentioned in the source text at all. "
    "If the text does not reference a bone marrow biopsy being performed, choose this."
)

CNS_MEANINGS = (
    "- \"Imaging abnormality\" — CNS imaging shows an abnormality but CSF is not mentioned.\n"
    "- \"CSF Positive\" — CSF analysis is positive for lymphoma cells.\n"
    "- \"Both\" — both imaging abnormality AND CSF positive are present.\n"
    "- \"Not assessed\" — CNS was NEVER evaluated: no imaging performed, no CSF analysis done. "
    "No mention of any CNS evaluation anywhere in the source text.\n"
    "- \"Not involved\" — CNS was evaluated AND found to be NORMAL/NOT involved. "
    "The source states findings like 'CNS WNL', 'CNS NAD', 'no CNS involvement', 'CNS normal'."
)

HARD_FIELDS = [
    {
        "title": "BM biopsy",
        "name_regex": r"mcq_1755701329003",
        "option_meanings": BM_BIOPSY_MEANINGS,
        "instructions": (
            "If the bone marrow biopsy is not referred to as having been performed "
            "in the medical report, choose \"Not done\"."
        ),
    },
    {
        "title": "CNS",
        "name_regex": r"mcq_1755701655242",
        "option_meanings": CNS_MEANINGS,
        "instructions": (
            "Carefully distinguish 'Not assessed' (no CNS evaluation mentioned at all) "
            "from 'Not involved' (CNS was evaluated and results were normal/negative). "
            "Look for phrases like 'CNS WNL', 'CNS NAD', 'no CNS involvement'."
        ),
    },
]


def _extract_source(input_text: str) -> str:
    if "Source text:\n" in input_text:
        return input_text.split("Source text:\n", 1)[1].strip()
    return ""


def _parse_draft(response: str) -> dict[str, Any]:
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            inner = data.get("final_answer", data)
            if isinstance(inner, dict):
                return inner
            if isinstance(inner, str) and inner:
                try:
                    return json.loads(inner)
                except json.JSONDecodeError:
                    pass
        return data
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(extract_json_field(response, "final_answer"))
    except (json.JSONDecodeError, Exception):
        return {}


def _apply_correction(
    draft: dict[str, Any], field_cfg: dict[str, Any], corrected: str, field_name: str
) -> None:
    if corrected and corrected.lower() != "null":
        draft[field_name] = {"answer": corrected.strip().strip('"').strip("'")}
    else:
        draft[field_name] = {"answer": ""}


def predict(input: str, call_llm) -> tuple[str, dict]:
    source_text = _extract_source(input)

    draft_response = call_llm(FILL_PROMPT.format(input=input))
    draft = _parse_draft(draft_response)

    reviewed = []
    for field_cfg in HARD_FIELDS:
        field_name = None
        for key in draft:
            if re.search(field_cfg["name_regex"], key):
                field_name = key
                break
        if not field_name:
            continue

        review_prompt = BLIND_REVIEW_PROMPT.format(
            field_title=field_cfg["title"],
            option_meanings=field_cfg["option_meanings"],
            instructions=field_cfg["instructions"],
            source_text=source_text,
        )

        review_response = call_llm(review_prompt)
        corrected = review_response.strip().strip('"').strip("'")

        _apply_correction(draft, field_cfg, corrected, field_name)
        reviewed.append(field_cfg["title"])

    return draft, {
        "strategy": "blind_verifier",
        "reviewed_fields": reviewed,
    }