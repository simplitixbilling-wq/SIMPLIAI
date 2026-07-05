import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app_core.python_analysis_runner import PythonAnalysisRunner


class PythonAnalysisRunnerTests(unittest.TestCase):
    def make_runner(self, output_dir, logs=None):
        log_messages = logs if logs is not None else []
        return PythonAnalysisRunner(
            output_dir=output_dir,
            log=log_messages.append,
            time_fn=lambda: 1000.0,
        )

    def test_run_executes_code_and_saves_dataframe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(temp_dir)
            dataframes = {"df": pd.DataFrame({"amount": [1, 2, 3]})}

            result = runner.run(
                "result_df = df[df['amount'] > 1]",
                dataframes=dataframes,
                output_format="csv",
                status_messages=["[PYTHON] Completed"],
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["row_count"], 2)
            self.assertEqual(result["col_count"], 1)
            self.assertEqual(Path(result["file_path"]).name, "analysis_1000.csv")
            self.assertTrue(Path(result["file_path"]).exists())
            self.assertIn("Total rows: 2", result["response_text"])

    def test_run_rejects_missing_result_df(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(temp_dir)

            result = runner.run(
                "answer = df.head()",
                dataframes={"df": pd.DataFrame({"a": [1]})},
                output_format="csv",
                status_messages=[],
            )

            self.assertIn("result_df", result["error"])

    def test_run_rejects_non_dataframe_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(temp_dir)

            result = runner.run(
                "result_df = 123",
                dataframes={"df": pd.DataFrame({"a": [1]})},
                output_format="csv",
                status_messages=[],
            )

            self.assertIn("not a DataFrame", result["error"])

    def test_run_reports_execution_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(temp_dir)

            result = runner.run(
                "result_df = df['missing']",
                dataframes={"df": pd.DataFrame({"a": [1]})},
                output_format="csv",
                status_messages=[],
            )

            self.assertTrue(result["error"].startswith("Python execution failed"))
            self.assertIn("missing", result["exception"])


if __name__ == "__main__":
    unittest.main()
