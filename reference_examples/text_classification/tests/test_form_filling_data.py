import json
import tempfile
import unittest
from pathlib import Path

from reference_examples.text_classification.form_filling_data import (
    _form_template_from_schema,
    _output_state_skeleton,
    check_answer,
    load_dataset_splits_3way,
)


REALISTIC_SCHEMA = [
    {
        "component": "FormMCQField",
        "title": "Staging",
        "name": "mcq_stage",
        "config": {
            "question": "Is staging evaluation of cancer done?",
            "options": ["Yes", "No"],
            "multipleChoice": False,
        },
    },
    {
        "component": "FormMCQField",
        "title": "Staging method",
        "name": "mcq_stage_method",
        "config": {
            "question": "If yes method used for staging evaluation",
            "options": [
                "PET-CT (Positron emission tomography-computerized tomography)",
                "CECT (Contrast enhanced computerized tomography)",
            ],
            "multipleChoice": False,
        },
        "condition": {
            "field": "mcq_stage",
            "operator": "equals",
            "value": "Yes",
        },
    },
]


class FormFillingDataTests(unittest.TestCase):
    def test_load_dataset_formats_prompt_and_scores_exact_json(self):
        train, val, test, evaluator = load_dataset_splits_3way("FormFilling", 1, 0, 0)

        self.assertEqual((len(train), len(val), len(test)), (1, 0, 0))
        self.assertIn("Fill the form from the source text.", train[0]["input"])
        self.assertIn("Input form template for validation:", train[0]["input"])
        self.assertIn("Expected output state shape:", train[0]["input"])
        self.assertEqual(len(train[0]["fields"]), 67)
        self.assertEqual(
            train[0]["fields"][:3],
            ["input_name", "input_ipd_no", "input_phone_no"],
        )

        result = evaluator(
            train[0]["target"],
            train[0]["target"],
            fields=train[0]["fields"],
            form_template=train[0]["form_template"],
        )

        self.assertTrue(result["was_correct"])
        self.assertEqual(result["metrics"]["field_accuracy"], 1.0)

    def test_scores_wrapped_field_value_lists(self):
        prediction = {
            "answers": [
                {"name": "mcq_stage", "value": "Yes"},
                {
                    "name": "mcq_stage_method",
                    "value": "PET-CT (Positron emission tomography-computerized tomography)",
                },
            ]
        }
        target = {
            "Staging": "Yes",
            "Staging method": "PET-CT",
        }

        result = check_answer(
            prediction,
            target,
            form_template=REALISTIC_SCHEMA,
        )

        self.assertTrue(result["was_correct"])
        self.assertEqual(result["metrics"]["correct_fields"], 2)
        self.assertEqual(result["metrics"]["total_fields"], 2)

    def test_partial_targets_do_not_force_every_schema_field(self):
        result = check_answer(
            {"Is staging evaluation of cancer done?": "No"},
            {"Staging": "No"},
            form_template=REALISTIC_SCHEMA,
        )

        self.assertTrue(result["was_correct"])
        self.assertEqual(result["metrics"]["field_results"], {"mcq_stage": True})
        self.assertEqual(result["metrics"]["missing_fields"], 0)

    def test_nested_filled_form_builder_output_is_flattened(self):
        target = {
            "group_basic_information": {
                "input_name": "john",
                "select_mother_referred": "No",
            },
            "group_pregnancy_care": {
                "table_obstetric_history": [
                    [{"editable": False, "value": "Gravida"}],
                    [{"editable": True, "value": "1"}],
                ],
                "group_past_history": {
                    "select_past_history_anemia": "N",
                },
            },
            "group_examination_labour_birth": {
                "mcq_visible_birth_defect_type": {"answer": "Club foot"},
            },
            "group_other_associated_conditions": {
                "mcq_critical_delays": {
                    "answer": ["Delay in seeking care", "Delay in reaching care"]
                },
            },
        }
        prediction = {
            "input_name": "john",
            "select_mother_referred": "No",
            "table_obstetric_history": [["Gravida"], ["1"]],
            "select_past_history_anemia": "N",
            "mcq_visible_birth_defect_type": "Club foot",
            "mcq_critical_delays": ["Delay in reaching care", "Delay in seeking care"],
        }

        result = check_answer(prediction, target)

        self.assertTrue(result["was_correct"])
        self.assertEqual(result["metrics"]["correct_fields"], 6)
        self.assertEqual(result["metrics"]["total_fields"], 6)

    def test_jsonl_rows_can_have_different_schemas(self):
        rows = [
            {
                "input": "Transcript says alpha is yes.",
                "schema": [
                    {
                        "component": "FormMCQField",
                        "title": "Alpha",
                        "name": "field_alpha",
                        "config": {
                            "question": "Is alpha present?",
                            "options": ["Yes", "No"],
                        },
                    }
                ],
                "target": {"field_alpha": "Yes"},
            },
            {
                "input": "Transcript says beta is no.",
                "schema": [
                    {
                        "component": "FormMCQField",
                        "title": "Beta",
                        "name": "field_beta",
                        "config": {
                            "question": "Is beta present?",
                            "options": ["Yes", "No"],
                        },
                    }
                ],
                "target": {"field_beta": "No"},
            },
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            dataset_path = Path(f.name)
            for row in rows:
                f.write(json.dumps(row) + "\n")

        try:
            train, val, test, evaluator = load_dataset_splits_3way(
                str(dataset_path), 1, 1, 0, shuffle_seed=0
            )
        finally:
            dataset_path.unlink(missing_ok=True)

        examples = train + val + test
        self.assertEqual(
            {tuple(ex["fields"]) for ex in examples},
            {("field_alpha",), ("field_beta",)},
        )
        for ex in examples:
            self.assertTrue(ex["schema_is_row_specific"])
            self.assertEqual(ex["raw_question"], ex["input"])
            self.assertIn("Input form template for validation:", ex["input"])
            self.assertIn("Expected output state shape:", ex["input"])
            if ex["fields"] == ["field_alpha"]:
                self.assertIn("Alpha", ex["input"])
                self.assertNotIn("Beta", ex["input"])
                result = evaluator(
                    {"field_alpha": "Yes"},
                    {"field_alpha": "Yes"},
                    fields=ex["fields"],
                    form_template=ex["form_template"],
                )
                self.assertTrue(result["was_correct"])

    def test_jsonl_manifest_can_reference_input_schema_and_target_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "transcripts").mkdir()
            (base / "schemas").mkdir()
            (base / "targets").mkdir()

            (base / "transcripts" / "case_001.txt").write_text(
                "Transcript says the patient name is Jane and alpha is yes."
            )
            schema = [
                {
                    "component": "FormGroup",
                    "title": "Basic",
                    "name": "group_basic_information",
                    "fields": [
                        {
                            "component": "FormInputField",
                            "title": "Name",
                            "name": "input_name",
                            "type": "text",
                        },
                        {
                            "component": "FormMCQField",
                            "title": "Alpha",
                            "name": "mcq_alpha",
                            "config": {
                                "question": "Is alpha present?",
                                "options": ["Yes", "No"],
                                "multipleChoice": False,
                            },
                        },
                    ],
                }
            ]
            target = {
                "group_basic_information": {
                    "input_name": "Jane",
                    "mcq_alpha": {"answer": "Yes"},
                }
            }
            (base / "schemas" / "form.json").write_text(json.dumps(schema))
            (base / "targets" / "case_001.filled.json").write_text(
                json.dumps(target)
            )
            manifest = base / "form_filling.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "input_path": "transcripts/case_001.txt",
                        "schema_path": "schemas/form.json",
                        "target_path": "targets/case_001.filled.json",
                    }
                )
                + "\n"
            )

            train, val, test, evaluator = load_dataset_splits_3way(
                str(manifest), 1, 0, 0, shuffle_seed=0
            )

        self.assertEqual((len(train), len(val), len(test)), (1, 0, 0))
        example = train[0]
        self.assertIn("patient name is Jane", example["input"])
        self.assertIn("Original form template JSON:", example["input"])
        self.assertEqual(example["fields"], ["input_name", "mcq_alpha"])
        self.assertEqual(example["raw_form_template"], schema)
        self.assertTrue(example["schema_is_row_specific"])
        self.assertEqual(
            example["raw_input"],
            "Transcript says the patient name is Jane and alpha is yes.",
        )

        result = evaluator(
            target,
            example["target"],
            fields=example["fields"],
            form_template=example["form_template"],
        )
        self.assertTrue(result["was_correct"])

    def test_filled_form_can_seed_schema_fields(self):
        filled_form = {
            "group_basic_information": {
                "input_name": "john",
                "select_mother_referred": "No",
            },
            "group_other_associated_conditions": {
                "mcq_critical_delays": {
                    "answer": ["Delay in seeking care"],
                },
            },
        }

        template = _form_template_from_schema(filled_form)

        self.assertEqual(
            [field["name"] for field in template],
            ["input_name", "mcq_critical_delays", "select_mother_referred"],
        )

    def test_grouped_template_schema_is_flattened(self):
        schema = [
            {
                "component": "FormGroup",
                "title": "1. Basic Information",
                "name": "group_basic_information",
                "fields": [
                    {
                        "component": "FormInputField",
                        "title": "Name",
                        "name": "input_name",
                        "type": "text",
                    },
                    {
                        "component": "FormSelectField",
                        "title": "Mother referred",
                        "name": "select_mother_referred",
                        "options": [
                            {"value": "", "label": "Select..."},
                            {"value": "Yes", "label": "Yes"},
                            {"value": "No", "label": "No"},
                        ],
                    },
                    {
                        "component": "FormTableField",
                        "title": "Obstetric history",
                        "name": "table_obstetric_history",
                        "options": {
                            "cells": [
                                [
                                    {"editable": False, "value": "Gravida"},
                                    {"editable": False, "value": "Para"},
                                ],
                                [
                                    {"editable": True, "value": ""},
                                    {"editable": True, "value": ""},
                                ],
                            ]
                        },
                    },
                ],
            }
        ]

        template = _form_template_from_schema(schema)

        self.assertEqual(
            [field["name"] for field in template],
            ["input_name", "select_mother_referred", "table_obstetric_history"],
        )
        self.assertEqual(template[0]["group"], "1. Basic Information")
        self.assertEqual(template[0]["group_path"], ["group_basic_information"])
        self.assertEqual(template[1]["type"], "select")
        self.assertEqual(template[1]["options"], ["Yes", "No"])
        self.assertEqual(
            template[2]["table"],
            [["Gravida", "Para"], ["", ""]],
        )
        self.assertEqual(
            _output_state_skeleton(template),
            {
                "group_basic_information": {
                    "input_name": "",
                    "select_mother_referred": "",
                    "table_obstetric_history": [
                        [
                            {"editable": False, "value": "Gravida"},
                            {"editable": False, "value": "Para"},
                        ],
                        [
                            {"editable": True, "value": ""},
                            {"editable": True, "value": ""},
                        ],
                    ],
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
