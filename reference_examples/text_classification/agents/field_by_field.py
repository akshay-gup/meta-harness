"""Field-by-field no-memory baseline.

This is intentionally close to no_memory, but fills one field at a time and
maintains an in-example form state so conditional and related fields can see
answers already produced for earlier fields.
"""

import copy
import json
import re
from typing import Any

from ..form_filling_data import _form_template_from_schema, _output_state_skeleton
from ..llm import LLMCallable
from ..memory_system import MemorySystem


FIELD_PROMPT = """Fill exactly one field in a larger form.

Use the source text as evidence. Use the current completed state to preserve
cross-field context and to evaluate conditional fields.

Current completed state so far:
{state}

Field to fill:
{field}

Rules:
- Return only JSON in the shape {{"value": ...}}.
- For single-choice fields, return exactly one allowed option string.
- For multiple-choice fields, return a JSON array of allowed option strings.
- For text/number fields, return a concise string copied or normalized from the source.
- If the field is not supported by the source, return an empty string.
- If a condition is not met according to the current completed state, return an empty string.
- Do not fill or revise any other field.

Source text:
{source}
"""


EMPTY_STRINGS = {
    "",
    "null",
    "none",
    "n/a",
    "na",
    "not found",
    "not present",
    "not specified",
    "unknown",
}


class FieldByField(MemorySystem):
    """No-learning baseline that fills a form sequentially, one field per call."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self._state = "{}"

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        fields = _extract_fields(input)
        source = _extract_source(input)
        skeleton = _extract_output_skeleton(input)

        output = copy.deepcopy(skeleton) if isinstance(skeleton, dict) else {}
        if not output:
            output = _output_state_skeleton(fields)
        if not fields or not source:
            return json.dumps(output, ensure_ascii=False), {
                "strategy": "field_by_field",
                "num_fields": len(fields),
                "llm_calls": 0,
                "skipped_conditions": 0,
                "parse_error": True,
            }

        fields_by_name = {field["name"]: field for field in fields if field.get("name")}
        state_by_name: dict[str, Any] = {}
        llm_calls = 0
        skipped_conditions = 0

        for field in fields:
            name = field.get("name")
            if not name:
                continue

            condition_status = _condition_status(field.get("condition"), state_by_name)
            if condition_status is False:
                value = _empty_value(field)
                skipped_conditions += 1
            else:
                field_prompt = FIELD_PROMPT.format(
                    state=_state_summary(state_by_name, fields_by_name),
                    field=_format_field(field, fields_by_name),
                    source=source,
                )
                response = self.call_llm(field_prompt)
                raw_value = _extract_field_value(response)
                value = _coerce_value(field, raw_value)
                llm_calls += 1

            _set_output_value(output, field, value)
            state_by_name[name] = value

        return json.dumps(output, ensure_ascii=False), {
            "strategy": "field_by_field",
            "num_fields": len(fields),
            "llm_calls": llm_calls,
            "skipped_conditions": skipped_conditions,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        """No learning; this baseline only uses per-example state."""
        pass

    def get_state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        self._state = state


def _extract_fields(input_text: str) -> list[dict[str, Any]]:
    raw_template = _extract_json_after(input_text, "Original form template JSON:")
    if raw_template is not None:
        fields = _form_template_from_schema(raw_template)
        if fields:
            return fields

    template_text = _extract_between(
        input_text,
        "Input form template for validation:",
        "Expected output state shape:",
    )
    return _parse_formatted_fields(template_text)


def _extract_output_skeleton(input_text: str) -> dict[str, Any] | None:
    value = _extract_json_after(input_text, "Expected output state shape:")
    return value if isinstance(value, dict) else None


def _extract_source(input_text: str) -> str:
    marker = "Source text:\n"
    if marker not in input_text:
        return ""
    source = input_text.split(marker, 1)[1]
    for stop in (
        "\n\nReturn the completed form state",
        "\n\nReturn the completed",
        "\n\nProduce the expected output",
    ):
        if stop in source:
            source = source.split(stop, 1)[0]
    return source.strip()


def _extract_between(text: str, start_marker: str, end_marker: str) -> str:
    if start_marker not in text:
        return ""
    chunk = text.split(start_marker, 1)[1]
    if end_marker in chunk:
        chunk = chunk.split(end_marker, 1)[0]
    return chunk.strip()


def _extract_json_after(text: str, marker: str) -> Any:
    if marker not in text:
        return None
    return _extract_first_json(text.split(marker, 1)[1])


def _extract_first_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text):
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def _parse_formatted_fields(template_text: str) -> list[dict[str, Any]]:
    fields = []
    blocks = re.split(r"\n(?=\d+\.\s)", template_text.strip())
    for block in blocks:
        header = re.search(r"^\d+\.\s+(.+?)\s*(?:\u2014|\u2013|-)\s*(.+)$", block, re.M)
        if not header:
            continue
        field: dict[str, Any] = {
            "name": header.group(1).strip(),
            "title": header.group(2).strip(),
            "question": header.group(2).strip(),
            "component": "FormInputField",
            "type": "text",
            "options": [],
            "multipleChoice": False,
            "group_path": [],
        }
        for line in block.splitlines()[1:]:
            line = line.strip()
            if line.startswith("Type:"):
                field["type"] = line.split(":", 1)[1].strip()
                if field["type"] in {"mcq", "select"}:
                    field["component"] = "FormMCQField"
            elif line.startswith("Question:"):
                field["question"] = line.split(":", 1)[1].strip()
            elif line.startswith("Options:"):
                options = line.split(":", 1)[1].split(";")
                field["options"] = [option.strip() for option in options if option.strip()]
            elif line == "Multiple answers allowed":
                field["multipleChoice"] = True
            elif line.startswith("Instructions:"):
                field["instructions"] = line.split(":", 1)[1].strip()
            elif line.startswith("Only fill when "):
                field["condition"] = _parse_condition_line(line)
        fields.append(field)
    return fields


def _parse_condition_line(line: str) -> dict[str, Any] | None:
    match = re.search(r"\(([^)]+)\)\s+(\S+)\s+(.+)$", line)
    if not match:
        return None
    return {
        "field": match.group(1).strip(),
        "operator": match.group(2).strip(),
        "value": match.group(3).strip(),
    }


def _format_field(
    field: dict[str, Any],
    fields_by_name: dict[str, dict[str, Any]],
) -> str:
    parts = [
        f"Name: {field.get('name', '')}",
        f"Title: {field.get('title') or field.get('name', '')}",
        f"Type: {field.get('type', 'text')}",
    ]
    question = field.get("question")
    if question:
        parts.append(f"Question: {question}")
    options = field.get("options") or []
    if options:
        parts.append("Allowed options: " + json.dumps(options, ensure_ascii=False))
    if field.get("multipleChoice"):
        parts.append("Multiple answers allowed: true")
    condition = field.get("condition")
    if isinstance(condition, dict):
        parent_name = condition.get("field", "")
        parent = fields_by_name.get(parent_name, {})
        parent_title = parent.get("title") or parent_name
        parts.append(
            "Condition: only fill when "
            f"{parent_title} ({parent_name}) "
            f"{condition.get('operator', 'equals')} {condition.get('value')}"
        )
    instructions = field.get("instructions")
    if instructions:
        parts.append(f"Additional instructions: {instructions}")
    return "\n".join(parts)


def _state_summary(
    state_by_name: dict[str, Any],
    fields_by_name: dict[str, dict[str, Any]],
) -> str:
    summary = {}
    for name, value in state_by_name.items():
        field = fields_by_name.get(name, {})
        label = field.get("title") or name
        summary[f"{label} ({name})"] = _unwrap_value(value)
    return json.dumps(summary, ensure_ascii=False, indent=2)


def _extract_field_value(response: str) -> Any:
    parsed = _extract_first_json(response)
    if isinstance(parsed, dict):
        for key in ("value", "answer", "final_answer"):
            if key in parsed:
                return parsed[key]
    if parsed is not None:
        return parsed
    return response.strip()


def _coerce_value(field: dict[str, Any], raw: Any) -> Any:
    raw = _unwrap_value(raw)
    if _is_empty(raw):
        return _empty_value(field)

    if _is_choice_field(field):
        if field.get("multipleChoice"):
            raw_items = raw if isinstance(raw, list) else _split_choice_items(str(raw))
            matched = [_match_option(item, field.get("options") or []) for item in raw_items]
            return {"answer": [item for item in matched if not _is_empty(item)]}
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        return {"answer": _match_option(str(raw), field.get("options") or [])}

    if field.get("type") == "table":
        return raw
    return str(raw).strip()


def _empty_value(field: dict[str, Any]) -> Any:
    if _is_choice_field(field):
        return {"answer": [] if field.get("multipleChoice") else ""}
    if field.get("type") == "table":
        return field.get("table_cells") or field.get("table") or []
    return ""


def _is_choice_field(field: dict[str, Any]) -> bool:
    return field.get("component") in {"FormMCQField", "FormSelectField"} or field.get(
        "type"
    ) in {"mcq", "select"}


def _split_choice_items(text: str) -> list[str]:
    parsed = _extract_first_json(text)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [item.strip() for item in re.split(r",|;|\n", text) if item.strip()]


def _match_option(value: Any, options: list[str]) -> str:
    text = str(value).strip().strip('"').strip("'")
    if not options:
        return text
    text_key = _key(text)
    for option in options:
        if text.strip() == option.strip() or text_key == _key(option):
            return option
    for option in options:
        option_key = _key(option)
        if text_key and (text_key in option_key or option_key in text_key):
            return option
    return text


def _condition_status(condition: Any, state_by_name: dict[str, Any]) -> bool | None:
    if not isinstance(condition, dict):
        return True
    parent_name = condition.get("field")
    if parent_name not in state_by_name:
        return None

    actual = _unwrap_value(state_by_name[parent_name])
    expected = condition.get("value")
    operator = str(condition.get("operator", "equals")).lower()

    actual_values = actual if isinstance(actual, list) else [actual]
    actual_keys = {_key(item) for item in actual_values if not _is_empty(item)}
    expected_key = _key(expected)

    if operator in {"equals", "==", "is"}:
        return expected_key in actual_keys
    if operator in {"notequals", "not_equals", "!=", "not"}:
        return expected_key not in actual_keys
    if operator in {"contains", "includes"}:
        return any(expected_key in actual_key for actual_key in actual_keys)
    return expected_key in actual_keys


def _set_output_value(output: dict[str, Any], field: dict[str, Any], value: Any) -> None:
    parent = output
    for group_name in field.get("group_path") or []:
        child = parent.setdefault(group_name, {})
        if not isinstance(child, dict):
            child = {}
            parent[group_name] = child
        parent = child
    parent[field["name"]] = value


def _unwrap_value(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("answer", "value", "selected", "selection", "final_answer"):
            if key in value:
                return value[key]
    return value


def _is_empty(value: Any) -> bool:
    value = _unwrap_value(value)
    if value is None:
        return True
    if isinstance(value, list):
        return len(value) == 0
    if isinstance(value, dict):
        return not value
    return str(value).strip().casefold() in EMPTY_STRINGS


def _key(value: Any) -> str:
    text = str(value).casefold().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)
