import unittest

import pandas as pd

from app_core.sql_analysis_runner import SQLAnalysisRunner


class FakeBridge:
    stop_generation_flag = False
    model = None

    def _normalize_column_names(self, columns):
        return columns

    def _create_duckdb_schema(self, _conn, _dataframes):
        return "schema"


class SQLAnalysisRunnerTests(unittest.TestCase):
    def test_execute_direct_sql_without_model(self):
        runner = SQLAnalysisRunner(FakeBridge())
        statuses = []

        result = runner.execute(
            {"df_sales": pd.DataFrame({"amount": [10, 20, 30]})},
            "SELECT SUM(amount) AS total_amount FROM df_sales",
            "none",
            pipeline_status_messages=statuses,
        )

        self.assertTrue(result["ok"])
        self.assertIsNone(result["file_path"])
        self.assertEqual(result["file_paths"], [])
        self.assertIn("Total rows: 1", result["response_text"])


if __name__ == "__main__":
    unittest.main()
