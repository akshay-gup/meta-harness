import unittest

from reference_examples.text_classification.inner_loop import InferenceOnlyAdapter


class InferenceOnlyAdapterTests(unittest.TestCase):
    def test_wraps_plain_predict_function_with_call_llm(self):
        calls = []

        def fake_llm(prompt):
            calls.append(prompt)
            return '{"final_answer": {"field": "value"}}'

        def predict(input, call_llm):
            response = call_llm(f"fill: {input}")
            return response, {"kind": "function"}

        adapter = InferenceOnlyAdapter(fake_llm, predict)

        prediction, metadata = adapter.predict("case text")

        self.assertEqual(prediction, '{"final_answer": {"field": "value"}}')
        self.assertEqual(metadata, {"kind": "function"})
        self.assertEqual(calls, ["fill: case text"])
        self.assertEqual(adapter.get_last_prompt_info()["prompt_text"], "fill: case text")
        self.assertEqual(adapter.get_state(), "{}")

    def test_wraps_plain_agent_class_with_call_llm(self):
        class Agent:
            def __init__(self, call_llm):
                self.call_llm = call_llm

            def predict(self, input):
                return self.call_llm(f"agent: {input}")

        adapter = InferenceOnlyAdapter(lambda prompt: "{}", Agent)

        prediction, metadata = adapter.predict("case text")

        self.assertEqual(prediction, "{}")
        self.assertEqual(metadata, {"interface": "inference_only"})
        self.assertEqual(
            adapter.get_last_prompt_info()["prompt_text"], "agent: case text"
        )

    def test_serializes_dict_predictions(self):
        def predict(input):
            return {"field": input}, {"kind": "dict"}

        adapter = InferenceOnlyAdapter(lambda prompt: "unused", predict)

        prediction, metadata = adapter.predict("value")

        self.assertEqual(prediction, '{"field": "value"}')
        self.assertEqual(metadata, {"kind": "dict"})


if __name__ == "__main__":
    unittest.main()
