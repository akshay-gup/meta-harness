"""Draft-then-Verify Hard-Field Agent — inference-only predictor.

Hypothesis: A second targeted LLM call that re-examines the hardest MCQ fields
(BM biopsy, CNS) with explicit option semantics guidance will reduce wrong_choice
errors more effectively than a single-pass fill, because the first pass provides
context for the source evidence and the second pass focuses on the ambiguous
option distinctions that diagnostics reveal as high-error.
"""

import json
import re
from typing import Any

from ..memory_system import extract_json_field


FULL_FORM_PROMPT = (
    "Fill the form from the source text.\n\n"
    "{input}\n\n"
    "Return the completed form state in `final_answer`. Use the output state shape "
    "provided in the input when present. Do not return the blank input template or "
    "template metadata such as component/title/config/options.\n\n"
    "**Answer in this exact JSON format:**\n"
    '{{"reasoning": "[brief reasoning]", "final_answer": {{"completed": "form state"}}}}\n'
)

HARD_FIELD_REVIEW_PROMPT = (
    "Review and potentially correct these specific fields in the completed form.\n\n"
    "Current filled form:\n"
    "{current_form}\n\n"
    "Source text:\n"
    "{source_text}\n\n"
    "Field to review: **{field_title}** ({field_name})\n\n"
    "Here are the allowed options and their meanings:\n"
    "{option_meanings}\n\n"
    "Instructions:\n"
    "{instructions}\n\n"
    "Based on the source text above, what is the CORRECT answer? "
    "Return ONLY the exact option text string. If the field should be empty, return null.\n\n"
    "Answer:"
)


def _extract_source_text(input_text: str) -> str:
    if "Source text:\n" in input_text:
        return input_text.split("Source text:\n", 1)[1].strip()
    return ""


def _normalize_value(val: Any) -> Any:
    if isinstance(val, str):
        return val.strip()
    return val


BM_BIOPSY_OPTION_MEANINGS = (
    "- \"Yes\" — use when the source explicitly states BM biopsy SHOWS lymphoma involvement.\n"
    "- \"No\" — use when the source states BM biopsy was done and is NEGATIVE/NO involvement.\n"
    "- \"Not done\" — use when no BM biopsy is mentioned at all in the source text. "
    "If the source does not reference a bone marrow biopsy, choose this."
)

CNS_OPTION_MEANINGS = (
    "- \"Imaging abnormality\" — CNS imaging shows abnormality (but CSF not mentioned).\n"
    "- \"CSF Positive\" — CSF analysis is positive for lymphoma cells.\n"
    "- \"Both\" — both imaging abnormality AND CSF positive are present.\n"
    "- \"Not assessed\" — CNS was NEVER evaluated (no imaging, no CSF). "
    "No mention of CNS evaluation anywhere in the source.\n"
    "- \"Not involved\" — CNS was evaluated and found to be NORMAL/NOT involved. "
    "The source states CNS findings are normal (e.g. CNS WNL, CNS NAD, CNS normal, no CNS involvement)."
)

# Field-specific review configuration
REVIEW_FIELDS = [
    {
        "title": "BM biopsy",
        "name_regex": r"mcq_1755701329003",
        "option_meanings": BM_BIOPSY_OPTION_MEANINGS,
        "instructions": "If the bone marrow biopsy is not referred to as done in the medical report, choose \"Not done\".",
    },
    {
        "title": "CNS",
        "name_regex": r"mcq_1755701655242",
        "option_meanings": CNS_OPTION_MEANINGS,
        "instructions": "Carefully distinguish 'Not assessed' (no CNS evaluation mentioned) from 'Not involved' (CNS evaluated and normal/NOT involved).",
    },
]


def predict(input: str, call_llm) -> tuple[str, dict]:
    source_text = _extract_source_text(input)

    # Phase 1: Full form draft
    response_1 = call_llm(FULL_FORM_PROMPT.format(input=input))

    draft = {}
    try:
        data = json.loads(response_1)
        if isinstance(data, dict):
            inner = data.get("final_answer", data)
            if isinstance(inner, dict):
                draft = inner
            elif isinstance(inner, str) and inner:
                try:
                    draft = json.loads(inner)
                except json.JSONDecodeError:
                    pass
        if not draft:
            draft = data
    except json.JSONDecodeError:
        try:
            draft = json.loads(extract_json_field(response_1, "final_answer"))
        except (json.JSONDecodeError, Exception):
            pass

    # Phase 2: Review each hard field
    corrections = {}
    for field_cfg in REVIEW_FIELDS:
        field_name = None
        for key in draft:
            if re.search(field_cfg["name_regex"], key):
                field_name = key
                break
        if not field_name:
            continue

        current_val = draft.get(field_name, "")
        current_val_text = json.dumps(current_val, indent=2) if isinstance(current_val, (dict, list)) else str(current_val)

        review_prompt = HARD_FIELD_REVIEW_PROMPT.format(
            current_form=json.dumps(draft, indent=2, ensure_ascii=False),
            source_text=source_text,
            field_title=field_cfg["title"],
            field_name=field_name,
            option_meanings=field_cfg["option_meanings"],
            instructions=field_cfg["instructions"],
        )

        review_response = call_llm(review_prompt)
        corrected = review_response.strip().strip('"').strip("'")

        if corrected and corrected.lower() != "null":
            # Apply correction in proper wrapper format
            draft[field_name] = {"answer": corrected}
        else:
            draft[field_name] = {"answer": ""}

    return draft, {
        "strategy": "draft_verifier",
        "reviewed_fields": [f["title"] for f in REVIEW_FIELDS],
    }