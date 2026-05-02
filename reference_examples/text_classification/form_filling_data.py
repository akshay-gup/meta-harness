"""Dataset loading for the form-filling benchmark.

This module provides the small interface expected by inner_loop.py:

- ALL_TASKS
- load_dataset_splits()
- load_dataset_splits_3way()

For a form-filling benchmark, put examples in data/form_filling.jsonl. Each
line should contain either:

    {"input": "source text", "target": {"field": "value"}}

or the aliases:

    {"text": "source text", "output": {"field": "value"}}

If schemas vary by example, include a row-level schema/template:

    {"input": "source text", "schema": [...], "target": {"field": "value"}}

Large examples may use paths relative to the JSONL file instead:

    {"input_path": "transcripts/case_001.txt",
     "schema_path": "schemas/stillbirth.json",
     "target_path": "targets/case_001.filled.json"}

Targets may also be plain strings; JSON object targets get stricter structured
comparison plus field-level metrics.
"""

from __future__ import annotations

import ast
import json
import random
import re
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"
FORM_FILLING_DATASET = "FormFilling"
ALL_TASKS = [FORM_FILLING_DATASET]

_INPUT_KEYS = ("input", "text", "source", "source_text", "prompt")
_TARGET_KEYS = ("target", "output", "expected", "label")
_SCHEMA_KEYS = ("schema", "form_schema", "template", "form_template", "fields")
_INPUT_PATH_KEYS = (
    "input_path",
    "text_path",
    "source_path",
    "source_text_path",
    "prompt_path",
)
_TARGET_PATH_KEYS = ("target_path", "output_path", "expected_path", "label_path")
_SCHEMA_PATH_KEYS = (
    "schema_path",
    "form_schema_path",
    "template_path",
    "form_template_path",
    "fields_path",
)


def _dataset_path(dataset: str) -> Path:
    """Resolve a dataset name or file path to a JSONL file."""
    candidate = Path(dataset)
    if candidate.exists():
        return candidate

    if dataset == FORM_FILLING_DATASET:
        return DATA_DIR / "form_filling.jsonl"

    slug = re.sub(r"(?<!^)(?=[A-Z])", "_", dataset).lower()
    return DATA_DIR / f"{slug}.jsonl"


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    raise ValueError(f"Row missing one of required keys: {', '.join(keys)}")


def _first_optional(row: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _resolve_data_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def _read_data_path(value: Any, base_dir: Path, parse_json: bool) -> Any:
    path = _resolve_data_path(value, base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Referenced data file not found: {path}")

    text = path.read_text()
    should_parse_json = parse_json or path.suffix.lower() == ".json"
    if should_parse_json:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON file at {path}: {exc}") from exc
    return text


def _row_value(
    row: dict[str, Any],
    inline_keys: tuple[str, ...],
    path_keys: tuple[str, ...],
    base_dir: Path,
    parse_path_json: bool,
) -> Any:
    inline = _first_optional(row, inline_keys)
    if inline is not None:
        return inline

    path_value = _first_optional(row, path_keys)
    if path_value is not None:
        return _read_data_path(path_value, base_dir, parse_path_json)

    expected_keys = ", ".join(inline_keys + path_keys)
    raise ValueError(f"Row missing one of required keys: {expected_keys}")


def _row_optional_value(
    row: dict[str, Any],
    inline_keys: tuple[str, ...],
    path_keys: tuple[str, ...],
    base_dir: Path,
    parse_path_json: bool,
) -> Any | None:
    inline = _first_optional(row, inline_keys)
    if inline is not None:
        return inline

    path_value = _first_optional(row, path_keys)
    if path_value is not None:
        return _read_data_path(path_value, base_dir, parse_path_json)
    return None


def _loads_jsonish(value: Any) -> Any:
    """Parse common JSON-like model outputs while preserving raw strings."""
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return ""

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Handles responses with prose around a JSON object.
    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth, pos, in_str, escape = 1, start + 1, False, False
        while pos < len(text) and depth > 0:
            char = text[pos]
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_str = not in_str
            elif not in_str:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
            pos += 1
        if depth == 0:
            try:
                return json.loads(text[start:pos])
            except json.JSONDecodeError:
                pass

    # Handles Python reprs like {'name': 'Ada'} if a model/extractor produced one.
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return value.strip()


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return value


def _normalize(value: Any) -> Any:
    value = _loads_jsonish(value)
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return _normalize_scalar(value)


def _field_key(field: Any) -> str:
    """Canonical field key for loose matching of JSON and Markdown labels."""
    return re.sub(r"[^a-z0-9]", "", str(field).lower())


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _candidate_value_keys(item: dict[str, Any]) -> tuple[Any, bool]:
    for key in ("value", "answer", "response", "selected", "selection", "output"):
        if key in item:
            return item[key], True
    config = item.get("config")
    if isinstance(config, dict):
        for key in ("value", "answer", "response", "selected", "selection"):
            if key in config:
                return config[key], True
    return None, False


def _coerce_field_value_items(value: Any) -> dict[str, Any] | None:
    """Coerce common completed-form list shapes into {field_name: value}."""
    parsed = _loads_jsonish(value)
    if not isinstance(parsed, list):
        return None

    extracted: dict[str, Any] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = (
            item.get("name")
            or item.get("field")
            or item.get("key")
            or item.get("id")
        )
        field_value, has_value = _candidate_value_keys(item)
        if name and has_value:
            extracted[str(name)] = field_value

    return extracted or None


def _simplify_table_cells(value: Any) -> Any:
    """Drop form-builder cell metadata and keep cell values for table answers."""
    if not isinstance(value, list):
        return value

    simplified = []
    changed = False
    for item in value:
        if isinstance(item, list):
            row = []
            row_changed = False
            for cell in item:
                if isinstance(cell, dict):
                    cell_value, has_value = _candidate_value_keys(cell)
                    if has_value:
                        row.append(_simplify_table_cells(cell_value))
                        row_changed = True
                        continue
                row.append(_simplify_table_cells(cell))
            simplified.append(row)
            changed = changed or row_changed
        elif isinstance(item, dict):
            item_value, has_value = _candidate_value_keys(item)
            if has_value:
                simplified.append(_simplify_table_cells(item_value))
                changed = True
            else:
                simplified.append(item)
        else:
            simplified.append(item)

    return simplified if changed else value


def _unwrap_form_value(value: Any) -> Any:
    """Return the actual answer from common form-builder value wrappers."""
    parsed = _loads_jsonish(value)

    if isinstance(parsed, dict):
        field_value, has_value = _candidate_value_keys(parsed)
        if has_value:
            return _unwrap_form_value(field_value)

    if isinstance(parsed, list):
        return _simplify_table_cells(parsed)

    return parsed


def _normalize_form_value(value: Any) -> Any:
    return _normalize(_unwrap_form_value(value))


def _is_scalar_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        not isinstance(item, (dict, list)) for item in value
    )


def _field_names(form_template: list[dict[str, Any]], fields: list[str]) -> list[str]:
    if form_template:
        return [field["name"] for field in form_template if field.get("name")]
    return fields


def _field_by_name(form_template: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {field["name"]: field for field in form_template if field.get("name")}


def _field_aliases(
    fields: list[str], form_template: list[dict[str, Any]] | None = None
) -> dict[str, str]:
    """Map normalized field labels/titles/questions to canonical field names."""
    aliases = {_field_key(field): field for field in fields}
    for field in form_template or []:
        name = field.get("name")
        if not name:
            continue
        for value in (
            name,
            field.get("title"),
            field.get("question"),
            field.get("label"),
        ):
            if value:
                aliases[_field_key(value)] = name
    return aliases


def _parse_key_value_form(
    text: str, fields: list[str], form_template: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Parse simple Markdown/plaintext key-value forms."""
    aliases = _field_aliases(fields, form_template)
    parsed: dict[str, Any] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or re.fullmatch(r"\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?", line):
            continue

        # Markdown table row: | field | value |
        if "|" in line:
            cells = [cell.strip().strip("`*") for cell in line.strip("|").split("|")]
            cells = [cell for cell in cells if cell]
            if len(cells) >= 2:
                key = _field_key(cells[0])
                if key in aliases:
                    parsed[aliases[key]] = cells[1]
                    continue

        # Bullets or plain lines: - field: value, **field** = value
        match = re.match(
            r"^\s*(?:[-*]\s*)?(?:\*\*)?([^:=]+?)(?:\*\*)?\s*[:=]\s*(.+?)\s*$",
            line,
        )
        if not match:
            continue
        key = _field_key(match.group(1).strip().strip("`*"))
        if key in aliases:
            parsed[aliases[key]] = match.group(2).strip().strip("`*")

    return parsed


def _flatten_form_fields(
    value: Any,
    fields: list[str],
    form_template: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Flatten nested form-builder groups into a field-name keyed dict."""
    aliases = _field_aliases(fields, form_template)
    require_known_fields = bool(fields or form_template)
    flattened: dict[str, Any] = {}

    def visit(obj: Any) -> bool:
        parsed = _loads_jsonish(obj)
        list_form = _coerce_field_value_items(parsed)
        if list_form is not None:
            parsed = list_form
        if not isinstance(parsed, dict):
            return False

        found_form_fields = False
        for raw_key, raw_val in parsed.items():
            key = str(raw_key)
            alias = aliases.get(_field_key(key))
            value = _unwrap_form_value(raw_val)

            if alias:
                flattened[alias] = _normalize(value)
                found_form_fields = True
                continue

            nested_found = False
            if isinstance(value, dict):
                nested_found = visit(value)
            elif isinstance(value, list):
                nested_items = _coerce_field_value_items(value)
                if nested_items is not None:
                    nested_found = visit(nested_items)

            if not require_known_fields and not nested_found:
                flattened[key] = _normalize(value)
                found_form_fields = True
            else:
                found_form_fields = found_form_fields or nested_found

        return found_form_fields

    visit(value)
    return flattened or None


def _coerce_form_dict(
    value: Any,
    fields: list[str],
    form_template: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Coerce JSON, wrapped JSON, or Markdown/key-value output into a form dict."""
    parsed = _loads_jsonish(value)
    list_form = _coerce_field_value_items(parsed)
    if list_form is not None:
        parsed = list_form

    if isinstance(parsed, dict):
        aliases = _field_aliases(fields, form_template)
        by_key = {_field_key(key): val for key, val in parsed.items()}
        extracted = {
            aliases[key]: _normalize_form_value(val)
            for key, val in by_key.items()
            if key in aliases
        }
        if extracted:
            return extracted

        for wrapper in (
            "final_answer",
            "answer",
            "output",
            "result",
            "form",
            "completed_form",
            "fields",
            "values",
            "answers",
            "responses",
        ):
            wrapped_key = _field_key(wrapper)
            if wrapped_key in by_key:
                wrapped = _coerce_form_dict(by_key[wrapped_key], fields, form_template)
                if wrapped:
                    return wrapped

        flattened = _flatten_form_fields(parsed, fields, form_template)
        if flattened:
            return flattened

        if not fields:
            return {str(key): _normalize(val) for key, val in parsed.items()}
        return None

    if isinstance(parsed, str) and fields:
        extracted = _parse_key_value_form(parsed, fields, form_template)
        return {key: _normalize(val) for key, val in extracted.items()} or None

    return None


def _choice_key(value: Any) -> str:
    return _field_key(value)


def _choice_matches(prediction: Any, target: Any, field: dict[str, Any]) -> bool:
    pred_key = _choice_key(prediction)
    target_key = _choice_key(target)
    if pred_key == target_key:
        return True
    if len(pred_key) >= 3 and pred_key in target_key:
        return True
    if len(target_key) >= 3 and target_key in pred_key:
        return True

    for option in field.get("options", []):
        option_key = _choice_key(option)
        pred_is_option = pred_key == option_key or (
            len(pred_key) >= 3 and pred_key in option_key
        )
        target_is_option = target_key == option_key or (
            len(target_key) >= 3 and target_key in option_key
        )
        if pred_is_option and target_is_option:
            return True
    return False


_UNIT_TO_CM = {
    "mm": 0.1,
    "millimeter": 0.1,
    "millimeters": 0.1,
    "cm": 1.0,
    "centimeter": 1.0,
    "centimeters": 1.0,
    "m": 100.0,
    "meter": 100.0,
    "meters": 100.0,
    "in": 2.54,
    "inch": 2.54,
    "inches": 2.54,
}


def _field_default_unit(field: dict[str, Any] | None) -> str | None:
    if not field:
        return None
    haystack = " ".join(
        str(field.get(key, "")) for key in ("name", "title", "question", "instructions")
    ).lower()
    if "cm" in haystack or "centimeter" in haystack:
        return "cm"
    if "mm" in haystack or "millimeter" in haystack:
        return "mm"
    if re.search(r"\bin\b|inch", haystack):
        return "in"
    return None


def _parse_number_with_unit(
    value: Any, default_unit: str | None = None
) -> tuple[float, str | None] | None:
    text = str(value).lower().replace(",", "")
    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(mm|millimeters?|cm|centimeters?|m|meters?|in|inches?)?",
        text,
    )
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2) or default_unit
    return number, unit


def _numeric_values_match(
    prediction: Any, target: Any, field: dict[str, Any] | None = None
) -> bool:
    default_unit = _field_default_unit(field)
    pred = _parse_number_with_unit(prediction, default_unit)
    tgt = _parse_number_with_unit(target, default_unit)
    if not pred or not tgt:
        return False

    pred_num, pred_unit = pred
    tgt_num, tgt_unit = tgt
    if pred_unit in _UNIT_TO_CM and tgt_unit in _UNIT_TO_CM:
        pred_num *= _UNIT_TO_CM[pred_unit]
        tgt_num *= _UNIT_TO_CM[tgt_unit]

    tolerance = max(0.01, abs(tgt_num) * 0.01)
    return abs(pred_num - tgt_num) <= tolerance


def _values_match(
    prediction: Any, target: Any, field: dict[str, Any] | None = None
) -> bool:
    pred_norm = _normalize_form_value(prediction)
    target_norm = _normalize_form_value(target)
    if pred_norm == target_norm:
        return True
    if _is_scalar_list(pred_norm) and _is_scalar_list(target_norm):
        return sorted(_field_key(item) for item in pred_norm) == sorted(
            _field_key(item) for item in target_norm
        )
    if isinstance(pred_norm, (dict, list)) or isinstance(target_norm, (dict, list)):
        return False
    pred_text = _clean_text(pred_norm)
    target_text = _clean_text(target_norm)
    if pred_text.casefold() == target_text.casefold():
        return True
    if field and field.get("options") and _choice_matches(pred_text, target_text, field):
        return True
    if _numeric_values_match(pred_text, target_text, field):
        return True
    return _field_key(pred_text) == _field_key(target_text)


def _target_to_text(target: Any) -> str:
    if isinstance(target, (dict, list)):
        return json.dumps(target, sort_keys=True, ensure_ascii=False)
    return str(target)


def _schema_from_targets(targets: list[Any]) -> list[str]:
    fields: set[str] = set()
    for target in targets:
        parsed = _flatten_form_fields(target, [], None)
        if parsed:
            fields.update(parsed)
    return sorted(fields)


def _looks_like_filled_form(value: Any) -> bool:
    parsed = _loads_jsonish(value)
    if isinstance(parsed, list):
        return any(_looks_like_filled_form(item) for item in parsed)
    if not isinstance(parsed, dict):
        return False

    if any(str(key).startswith("group_") for key in parsed):
        return True
    if any(key in parsed for key in ("answer", "value", "selected", "selection")):
        return True
    if "editable" in parsed and "value" in parsed:
        return True
    return any(_looks_like_filled_form(item) for item in parsed.values())


def _template_field_from_name(
    name: str, group: str | None = None, group_path: list[str] | None = None
) -> dict[str, Any]:
    return {
        "name": str(name),
        "title": str(name),
        "question": str(name),
        "component": "FormInputField",
        "type": "text",
        "options": [],
        "multipleChoice": False,
        "table": [],
        "table_cells": [],
        "group": group,
        "group_path": list(group_path or []),
    }


def _option_text(option: Any) -> str:
    if isinstance(option, dict):
        label = option.get("label")
        value = option.get("value")
        if value == "" and label:
            return ""
        if label and value and str(label) != str(value):
            return f"{_clean_text(label)} ({_clean_text(value)})"
        return _clean_text(label if label is not None else value)
    return _clean_text(option)


def _table_cells(component: dict[str, Any]) -> list[list[dict[str, Any]]]:
    options = component.get("options")
    if isinstance(options, dict):
        cells = options.get("cells")
        if isinstance(cells, list):
            return cells
    return []


def _table_template(component: dict[str, Any]) -> list[list[Any]]:
    return _simplify_table_cells(_table_cells(component))


def _template_field_from_component(
    component: dict[str, Any],
    group: str | None = None,
    group_path: list[str] | None = None,
) -> dict[str, Any]:
    config = component.get("config") or {}
    name = str(component.get("name") or component.get("title") or config.get("question"))
    title = _clean_text(component.get("title") or name)
    question = _clean_text(config.get("question") or title)
    component_type = component.get("component", "FormInputField")
    if component_type == "FormMCQField":
        field_type = "mcq"
    elif component_type == "FormSelectField":
        field_type = "select"
    elif component_type == "FormTableField":
        field_type = "table"
    else:
        field_type = component.get("type") or "text"
    raw_options = config.get("options", component.get("options", []))
    option_values = raw_options if isinstance(raw_options, list) else []
    return {
        "name": name,
        "title": title,
        "question": question,
        "component": component_type,
        "type": field_type,
        "options": [
            option
            for option in (_option_text(option) for option in option_values)
            if option
        ],
        "multipleChoice": bool(config.get("multipleChoice", False)),
        "condition": component.get("condition"),
        "instructions": _clean_text(component.get("additionalInstructions", "")),
        "required": bool((component.get("validation") or {}).get("required")),
        "table": _table_template(component),
        "table_cells": _table_cells(component),
        "group": group,
        "group_path": list(group_path or []),
    }


def _normalize_form_template(
    form_template: list[Any] | None,
    group: str | None = None,
    group_path: list[str] | None = None,
) -> list[dict[str, Any]]:
    normalized = []
    group_path = list(group_path or [])
    for item in form_template or []:
        if isinstance(item, str):
            normalized.append(_template_field_from_name(item, group, group_path))
        elif isinstance(item, dict) and item.get("component") == "FormGroup":
            child_group = _clean_text(item.get("title") or item.get("name") or group or "")
            child_group_path = group_path + [str(item.get("name") or child_group)]
            normalized.extend(
                _normalize_form_template(
                    item.get("fields"),
                    child_group,
                    child_group_path,
                )
            )
        elif isinstance(item, dict) and (
            "config" in item or item.get("component", "").startswith("Form")
        ):
            normalized.append(_template_field_from_component(item, group, group_path))
        elif isinstance(item, dict) and item.get("name"):
            field = _template_field_from_name(str(item["name"]), group, group_path)
            field.update(item)
            field.setdefault("options", [])
            field.setdefault("multipleChoice", False)
            field.setdefault("group", group)
            field.setdefault("group_path", list(group_path))
            normalized.append(field)
    return normalized


def _form_template_from_schema(
    raw_schema: Any,
    targets: list[Any] | None = None,
) -> list[dict[str, Any]]:
    raw_schema = _loads_jsonish(raw_schema)
    if raw_schema is None:
        return []
    if isinstance(raw_schema, list):
        return _normalize_form_template(raw_schema)
    if isinstance(raw_schema, dict):
        if isinstance(raw_schema.get("fields"), list):
            return _normalize_form_template(raw_schema["fields"])
        if isinstance(raw_schema.get("components"), list):
            return _normalize_form_template(raw_schema["components"])
        if isinstance(raw_schema.get("schema"), (list, dict)):
            return _form_template_from_schema(raw_schema["schema"], targets)
        if _looks_like_filled_form(raw_schema):
            return [
                _template_field_from_name(field)
                for field in _schema_from_targets([raw_schema])
            ]
        return [_template_field_from_name(str(key)) for key in raw_schema]
    if isinstance(raw_schema, str):
        return [_template_field_from_name(raw_schema)]
    if targets:
        return [
            _template_field_from_name(field) for field in _schema_from_targets(targets)
        ]
    return []


def _load_global_form_template(targets: list[Any]) -> list[dict[str, Any]]:
    schema_path = DATA_DIR / "form_schema.json"
    if schema_path.exists():
        return _form_template_from_schema(json.loads(schema_path.read_text()), targets)
    return [_template_field_from_name(field) for field in _schema_from_targets(targets)]


def _load_form_template(targets: list[Any]) -> list[dict[str, Any]]:
    return _load_global_form_template(targets)


def _load_schema(targets: list[Any]) -> list[str]:
    return _field_names(_load_form_template(targets), [])


def _format_field_for_prompt(
    field: dict[str, Any],
    idx: int,
    fields_by_name: dict[str, dict[str, Any]] | None = None,
) -> str:
    parts = [f"{idx}. {field['name']} — {field.get('title') or field['name']}"]
    if field.get("group"):
        parts.append(f"Section: {field['group']}")
    if field.get("type"):
        parts.append(f"Type: {field['type']}")
    if field.get("question") and field["question"] != field.get("title"):
        parts.append(f"Question: {field['question']}")
    if field.get("options"):
        parts.append("Options: " + "; ".join(field["options"]))
    if field.get("multipleChoice"):
        parts.append("Multiple answers allowed")
    if field.get("table"):
        parts.append(
            "Table template: "
            + json.dumps(field["table"], ensure_ascii=False, separators=(",", ":"))
        )
    if field.get("condition"):
        cond = field["condition"]
        parent = cond.get("field")
        parent_field = (fields_by_name or {}).get(parent, {})
        parent_label = parent_field.get("title") or parent
        parts.append(
            "Only fill when "
            f"{parent_label} ({parent}) {cond.get('operator', 'equals')} {cond.get('value')}"
        )
    if field.get("instructions"):
        parts.append(f"Instructions: {field['instructions']}")
    return "\n   ".join(parts)


def _empty_state_value(field: dict[str, Any]) -> Any:
    if field.get("component") == "FormMCQField":
        return {"answer": [] if field.get("multipleChoice") else ""}
    if field.get("type") == "table":
        if field.get("table_cells"):
            return json.loads(json.dumps(field["table_cells"]))
        return [
            [{"editable": True, "value": cell} for cell in row]
            for row in field.get("table", [])
        ]
    return ""


def _output_state_skeleton(form_template: list[dict[str, Any]]) -> dict[str, Any]:
    skeleton: dict[str, Any] = {}
    for field in form_template:
        name = field.get("name")
        if not name:
            continue
        parent = skeleton
        for group_name in field.get("group_path") or []:
            parent = parent.setdefault(group_name, {})
        parent[name] = _empty_state_value(field)
    return skeleton


def _format_form_input(
    source_text: Any,
    form_template: list[dict[str, Any]],
    raw_form_template: Any | None = None,
) -> str:
    if isinstance(source_text, (dict, list)):
        source_text = json.dumps(source_text, sort_keys=True, ensure_ascii=False)
    source_text = str(source_text).strip()

    if form_template:
        fields_by_name = _field_by_name(form_template)
        fields_block = "\n".join(
            _format_field_for_prompt(field, idx, fields_by_name)
            for idx, field in enumerate(form_template, 1)
        )
        output_skeleton = json.dumps(
            _output_state_skeleton(form_template),
            ensure_ascii=False,
            indent=2,
        )
        raw_template_block = ""
        if raw_form_template is not None:
            raw_template_block = (
                "\n\nOriginal form template JSON:\n"
                + json.dumps(raw_form_template, ensure_ascii=False, indent=2)
            )
        return (
            "Fill the form from the source text.\n\n"
            "Input form template for validation:\n"
            f"{fields_block}\n\n"
            "Expected output state shape:\n"
            f"{output_skeleton}"
            f"{raw_template_block}\n\n"
            "Source text:\n"
            f"{source_text}\n\n"
            "Return the completed form state, not the input template. Fill only the "
            "`value` or `answer` slots in the output state shape. Do not return "
            "`component`, `title`, `config`, `options`, or validation metadata."
        )

    return (
        "Produce the expected output for the source text.\n\n"
        "Source text:\n"
        f"{source_text}"
    )


def _condition_is_met(
    condition: dict[str, Any] | None,
    target_form: dict[str, Any],
    fields_by_name: dict[str, dict[str, Any]],
) -> bool:
    if not condition:
        return True
    parent = condition.get("field")
    operator = condition.get("operator", "equals")
    expected = condition.get("value")
    actual = target_form.get(parent)
    parent_spec = fields_by_name.get(parent, {})
    if operator == "equals":
        return _values_match(actual, expected, parent_spec)
    if operator == "not_equals":
        return not _values_match(actual, expected, parent_spec)
    return True


def _active_required_fields(
    target_form: dict[str, Any],
    fields: list[str],
    form_template: list[dict[str, Any]],
) -> list[str]:
    if not form_template:
        return fields or list(target_form)

    fields_by_name = _field_by_name(form_template)
    active = []
    for field in form_template:
        name = field.get("name")
        if not name:
            continue
        condition = field.get("condition")
        if condition and not _condition_is_met(condition, target_form, fields_by_name):
            # If an inactive field has a non-empty reference value, keep scoring it;
            # otherwise skip fields hidden by template conditions.
            if _is_empty_value(target_form.get(name)):
                continue
        if name not in target_form:
            continue
        active.append(name)
    return active


def _load_jsonl_examples(dataset: str) -> list[dict[str, Any]]:
    path = _dataset_path(dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Add JSONL examples or update config.yaml."
        )

    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc

    base_dir = path.parent
    raw_inputs = [
        _row_value(row, _INPUT_KEYS, _INPUT_PATH_KEYS, base_dir, parse_path_json=False)
        for row in rows
    ]
    targets = [
        _row_value(row, _TARGET_KEYS, _TARGET_PATH_KEYS, base_dir, parse_path_json=True)
        for row in rows
    ]
    row_schemas = [
        _row_optional_value(
            row, _SCHEMA_KEYS, _SCHEMA_PATH_KEYS, base_dir, parse_path_json=True
        )
        for row in rows
    ]
    global_form_template = (
        _load_global_form_template(targets)
        if any(schema is None for schema in row_schemas)
        else []
    )

    examples = []
    for raw_input, target, row_schema in zip(
        raw_inputs, targets, row_schemas, strict=True
    ):
        has_row_schema = row_schema is not None
        form_template = (
            _form_template_from_schema(row_schema, [target])
            if has_row_schema
            else global_form_template
        )
        if not form_template:
            form_template = [
                _template_field_from_name(field) for field in _schema_from_targets([target])
            ]
        fields = _field_names(form_template, [])
        formatted_input = _format_form_input(raw_input, form_template, row_schema)
        examples.append(
            {
                "input": formatted_input,
                "target": _target_to_text(target),
                "fields": fields,
                "form_template": form_template,
                "raw_form_template": row_schema,
                "raw_question": formatted_input if has_row_schema else str(raw_input),
                "raw_input": raw_input,
                "schema_is_row_specific": has_row_schema,
            }
        )
    return examples


def check_answer(
    prediction: Any,
    target: Any,
    fields: list[str] | None = None,
    form_template: list[dict[str, Any]] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return form correctness and field-level metrics when possible."""
    form_template = _normalize_form_template(form_template)
    fields = _field_names(form_template, list(fields or []))
    fields_by_name = _field_by_name(form_template)

    target_form = _coerce_form_dict(target, fields, form_template)
    if target_form and not fields:
        fields = list(target_form)
    pred_form = _coerce_form_dict(prediction, fields, form_template)

    pred_norm = pred_form if pred_form is not None else _normalize(prediction)
    target_norm = target_form if target_form is not None else _normalize(target)

    metrics: dict[str, Any] = {}

    if target_form is not None and pred_form is not None:
        required_fields = _active_required_fields(target_form, fields, form_template)
        pred_keys = set(pred_form)
        field_results = {
            key: key in pred_form
            and _values_match(pred_form[key], target_form.get(key), fields_by_name.get(key))
            for key in required_fields
        }
        correct_fields = sum(
            1
            for key in required_fields
            if field_results[key]
        )
        incorrect = sum(
            1
            for key in required_fields
            if key in pred_form and not field_results[key]
        )
        extra = len(pred_keys - set(required_fields))
        missing = len([key for key in required_fields if key not in pred_form])
        total_fields = len(required_fields)
        precision = (
            correct_fields / (correct_fields + incorrect + extra)
            if correct_fields + incorrect + extra
            else 0.0
        )
        recall = correct_fields / total_fields if total_fields else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        metrics = {
            "correct_fields": correct_fields,
            "total_fields": total_fields,
            "field_accuracy": correct_fields / total_fields if total_fields else 0.0,
            "tp": correct_fields,
            "fp": incorrect + extra,
            "fn": missing,
            "extra_fields": extra,
            "missing_fields": missing,
            "field_results": field_results,
            "f1": f1,
        }

        # First-pass form filling cares whether all required fields are correct;
        # extra fields are tracked above but do not fail correctness yet.
        was_correct = correct_fields == total_fields and missing == 0
    else:
        was_correct = pred_norm == target_norm

    return {"was_correct": was_correct, "metrics": metrics}


def _split_examples(
    examples: list[dict[str, Any]],
    num_train: int,
    num_val: int,
    num_test: int,
    shuffle_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(examples)
    random.Random(shuffle_seed).shuffle(shuffled)

    requested = num_train + num_val + num_test
    if requested > len(shuffled):
        raise ValueError(
            f"Requested {requested} examples but dataset only has {len(shuffled)}."
        )

    train_end = num_train
    val_end = train_end + num_val
    test_end = val_end + num_test
    return (
        shuffled[:train_end],
        shuffled[train_end:val_end],
        shuffled[val_end:test_end],
    )


def load_dataset_splits(
    dataset: str,
    num_train: int,
    num_test: int,
    shuffle_seed: int = 42,
):
    examples = _load_jsonl_examples(dataset)
    train, _, test = _split_examples(
        examples,
        num_train=num_train,
        num_val=0,
        num_test=num_test,
        shuffle_seed=shuffle_seed,
    )
    return train, test, check_answer


def load_dataset_splits_3way(
    dataset: str,
    num_train: int,
    num_val: int,
    num_test: int,
    shuffle_seed: int = 42,
):
    examples = _load_jsonl_examples(dataset)
    train, val, test = _split_examples(
        examples,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        shuffle_seed=shuffle_seed,
    )
    return train, val, test, check_answer
