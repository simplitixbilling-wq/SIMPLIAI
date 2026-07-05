import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import app_core.python_analysis_workflow as python_analysis_workflow
from app_core.python_analysis_workflow import PythonAnalysisWorkflowRunner


class FakeModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def create_completion(self, prompt, **_kwargs):
        self.prompts.append(prompt)
        text = self.outputs.pop(0)
        return {"choices": [{"text": text}]}


class FakeBridge:
    def __init__(self, model):
        self.model = model

    def _create_enhanced_schema(self, dataframes):
        return "\n".join(f"{name}: {list(df.columns)}" for name, df in dataframes.items())


class PythonAnalysisWorkflowRunnerTests(unittest.TestCase):
    def test_retries_validation_failure_then_executes_repaired_code(self):
        model = FakeModel([
            "answer = df.head(1)",
            "result_df = df[df['amount'] > 1]",
        ])
        bridge = FakeBridge(model)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(python_analysis_workflow, "app_data_path", lambda rel="": str(Path(temp_dir) / rel) if rel else temp_dir):
                result = PythonAnalysisWorkflowRunner(bridge, max_attempts=2).execute(
                    {"df": pd.DataFrame({"amount": [1, 2, 3]})},
                    "keep amounts above 1",
                    "csv",
                    pipeline_status_messages=[],
                )

        self.assertTrue(result["ok"])
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["repair_attempts"], 1)
        self.assertEqual(len(model.prompts), 2)
        self.assertIn("PREVIOUS ATTEMPT FAILED", model.prompts[1])

    def test_returns_attempt_history_after_repeated_failures(self):
        model = FakeModel(["answer = df", "print(df)"])
        bridge = FakeBridge(model)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(python_analysis_workflow, "app_data_path", lambda rel="": str(Path(temp_dir) / rel) if rel else temp_dir):
                result = PythonAnalysisWorkflowRunner(bridge, max_attempts=2).execute(
                    {"df": pd.DataFrame({"amount": [1]})},
                    "summarize",
                    "csv",
                    pipeline_status_messages=[],
                )

        self.assertIn("failed after 2 attempt", result["error"])
        self.assertEqual(len(result["attempt_errors"]), 2)


if __name__ == "__main__":
    unittest.main()
