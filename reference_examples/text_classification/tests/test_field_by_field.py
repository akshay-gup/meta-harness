import json
import unittest

from reference_examples.text_classification.agents.field_by_field import FieldByField
from reference_examples.text_classification.form_filling_data import (
    _format_form_input,
    _form_template_from_schema,
)


def _prompt_for_schema(raw_schema):
    form_template = _form_template_from_schema(raw_schema)
    return _format_form_input("Parent and child evidence live here.", form_template, raw_schema)


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("Unexpected LLM call")
        return self.responses.pop(0)


class FieldByFieldTests(unittest.TestCase):
    def test_skips_conditional_field_when_parent_answer_does_not_match(self):
        raw_schema = [
            {
                "component": "FormMCQField",
                "title": "Parent",
                "name": "parent",
                "config": {
                    "question": "Is the parent true?",
                    "options": ["Yes", "No"],
                    "multipleChoice": False,
                },
            },
            {
                "component": "FormInputField",
                "title": "Child",
                "name": "child",
                "config": {"question": "Only fill when parent is yes"},
                "condition": {"field": "parent", "operator": "equals", "value": "Yes"},
            },
        ]
        llm = ScriptedLLM(['{"value": "No"}'])

        prediction, meta = FieldByField(llm).predict(_prompt_for_schema(raw_schema))
        output = json.loads(prediction)

        self.assertEqual(output["parent"], {"answer": "No"})
        self.assertEqual(output["child"], "")
        self.assertEqual(meta["llm_calls"], 1)
        self.assertEqual(meta["skipped_conditions"], 1)

    def test_passes_completed_state_into_conditional_child_prompt(self):
        raw_schema = [
            {
                "component": "FormMCQField",
                "title": "Parent",
                "name": "parent",
                "config": {
                    "question": "Is the parent true?",
                    "options": ["Yes", "No"],
                    "multipleChoice": False,
                },
            },
            {
                "component": "FormInputField",
                "title": "Child",
                "name": "child",
                "config": {"question": "Only fill when parent is yes"},
                "condition": {"field": "parent", "operator": "equals", "value": "Yes"},
            },
        ]
        llm = ScriptedLLM(['{"value": "Yes"}', '{"value": "child evidence"}'])

        prediction, meta = FieldByField(llm).predict(_prompt_for_schema(raw_schema))
        output = json.loads(prediction)

        self.assertEqual(output["parent"], {"answer": "Yes"})
        self.assertEqual(output["child"], "child evidence")
        self.assertEqual(meta["llm_calls"], 2)
        self.assertIn('"Parent (parent)": "Yes"', llm.prompts[1])


if __name__ == "__main__":
    unittest.main()
