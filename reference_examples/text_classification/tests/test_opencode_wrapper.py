import json
import os
import unittest
from unittest.mock import patch

from reference_examples.text_classification import opencode_wrapper


class OpenCodeWrapperTests(unittest.TestCase):
    def test_parse_stream_events_matches_session_result_contract(self):
        stdout = "\n".join(
            json.dumps(event)
            for event in [
                {
                    "type": "text",
                    "sessionID": "sess_1",
                    "part": {"text": "Created candidates."},
                },
                {
                    "type": "tool_use",
                    "part": {
                        "tool": "read",
                        "callID": "call_read",
                        "state": {
                            "status": "completed",
                            "input": {"filePath": "/repo/agents/no_memory.py"},
                            "output": "line one\nline two",
                        },
                    },
                },
                {
                    "type": "tool_use",
                    "part": {
                        "tool": "edit",
                        "callID": "call_edit",
                        "state": {
                            "status": "completed",
                            "input": {
                                "filePath": "/repo/agents/candidate.py",
                                "oldString": "old",
                                "newString": "new\ncontent",
                            },
                            "output": "ok",
                        },
                    },
                },
                {
                    "type": "step_finish",
                    "part": {
                        "sessionID": "sess_1",
                        "tokens": {
                            "input": 100,
                            "output": 20,
                            "cache": {"read": 12, "write": 3},
                        },
                        "cost": 0.001,
                    },
                },
            ]
        )

        result = opencode_wrapper.parse_stream_events(
            stdout,
            prompt="propose",
            model="ollama/qwen3-coder",
            duration=1.5,
            exit_code=0,
            cwd="/repo",
        )

        self.assertEqual(result.text, "Created candidates.")
        self.assertEqual(result.session_id, "sess_1")
        self.assertEqual(result.tool_calls[0].name, "Read")
        self.assertEqual(result.tool_calls[0].input["file_path"], "/repo/agents/no_memory.py")
        self.assertEqual(result.files_read, {"agents/no_memory.py": {"reads": 1, "lines": 2}})
        self.assertEqual(result.files_written, {"agents/candidate.py": {"lines_written": 2}})
        self.assertEqual(result.token_usage["input_tokens"], 100)
        self.assertEqual(result.token_usage["output_tokens"], 20)
        self.assertEqual(result.token_usage["cache_read_input_tokens"], 12)
        self.assertEqual(result.token_usage["cache_creation_input_tokens"], 3)
        self.assertEqual(result.cost_usd, 0.001)

    def test_build_command_uses_env_model_for_claude_aliases(self):
        with patch.dict(
            os.environ,
            {"OPENCODE_WRAPPER_MODEL": "ollama/qwen3-coder"},
            clear=True,
        ):
            cmd = opencode_wrapper.build_command("hello", model="opus")

        self.assertEqual(cmd[:3], ["opencode", "run", "--format"])
        self.assertIn("--model", cmd)
        self.assertIn("ollama/qwen3-coder", cmd)
        self.assertEqual(cmd[-1], "hello")

    def test_build_command_defers_alias_to_opencode_config_without_env_model(self):
        with patch.dict(os.environ, {}, clear=True):
            cmd = opencode_wrapper.build_command("hello", model="opus")

        self.assertNotIn("--model", cmd)
        self.assertEqual(cmd[-1], "hello")


if __name__ == "__main__":
    unittest.main()
