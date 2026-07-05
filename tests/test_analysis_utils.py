import unittest
import tempfile
from pathlib import Path

import pandas as pd

from app_core.analysis_utils import (
    clean_analysis_summary,
    clean_generated_python_code,
    detect_unsafe_python_keyword,
    output_extension,
    save_analysis_dataframe,
    validate_generated_python_code,
)


class AnalysisUtilsTests(unittest.TestCase):
    def test_clean_generated_python_code_removes_markdown_and_explanations(self):
        raw = """Here is the code:
```python
1. First do this
result_df = df.groupby("Date").sum().reset_index()
```
"""

        cleaned = clean_generated_python_code(raw)

        self.assertEqual(cleaned, 'result_df = df.groupby("Date").sum().reset_index()')

    def test_validate_generated_python_code_accepts_result_dataframe_assignment(self):
        ok, error = validate_generated_python_code("result_df = df.head(10)")

        self.assertTrue(ok)
        self.assertEqual(error, "")

    def test_validate_generated_python_code_requires_result_df(self):
        ok, error = validate_generated_python_code("answer = df.head(10)")

        self.assertFalse(ok)
        self.assertIn("result_df", error)

    def test_validate_generated_python_code_blocks_file_reads_and_imports(self):
        cases = [
            "result_df = pd.read_csv('x.csv')",
            "import os\nresult_df = df",
            "import numpy as np\nresult_df = df",
            "result_df = df\nprint(result_df)",
            "result_df = df\nopen('x.txt')",
            "result_df = df\n__import__('os')",
        ]

        for code in cases:
            with self.subTest(code=code):
                ok, error = validate_generated_python_code(code)
                self.assertFalse(ok)
                self.assertIn("Invalid code", error)

    def test_validate_generated_python_code_blocks_ast_escape_patterns(self):
        cases = [
            "result_df = df.__class__",
            "result_df = getattr(df, 'head')()",
            "def helper():\n    return df\nresult_df = helper()",
            "with df as x:\n    result_df = x",
        ]

        for code in cases:
            with self.subTest(code=code):
                ok, error = validate_generated_python_code(code)
                self.assertFalse(ok)
                self.assertIn("Invalid code", error)

    def test_validate_generated_python_code_reports_syntax_error(self):
        ok, error = validate_generated_python_code("result_df =")

        self.assertFalse(ok)
        self.assertIn("syntax error", error)

    def test_detect_unsafe_python_keyword_finds_runtime_escape_terms(self):
        self.assertEqual(detect_unsafe_python_keyword("result_df = eval(x)"), "eval")
        self.assertEqual(detect_unsafe_python_keyword("result_df = df.head()"), "")

    def test_clean_analysis_summary_drops_debug_lines_and_prefixes(self):
        summary = clean_analysis_summary(
            [
                "[PYTHON] Running generated code...",
                "[DEBUG] namespace dump",
                "[SQL-DEBUG] clean_sql input",
                "[AGENT] Saved file",
            ],
            "Total rows: 5",
        )

        self.assertEqual(summary, "Running generated code...\nSaved file\n\nTotal rows: 5")

    def test_output_extension_maps_known_formats(self):
        self.assertEqual(output_extension("excel"), "xlsx")
        self.assertEqual(output_extension("csv_json"), "csv")
        self.assertEqual(output_extension("unknown", default="txt"), "txt")

    def test_save_analysis_dataframe_writes_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            df = pd.DataFrame({"name": ["alpha"], "amount": [10]})

            result = save_analysis_dataframe(df, temp_dir, "csv", timestamp=123)

            path = Path(result["file_path"])
            self.assertEqual(path.name, "analysis_123.csv")
            self.assertEqual(result["extension"], "csv")
            self.assertEqual(result["writer"], "csv")
            self.assertTrue(path.exists())
            self.assertIn("alpha", path.read_text(encoding="utf-8"))

    def test_save_analysis_dataframe_writes_excel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            df = pd.DataFrame({"name": ["alpha"], "amount": [10]})

            result = save_analysis_dataframe(df, temp_dir, "excel", timestamp=321)

            path = Path(result["file_path"])
            self.assertEqual(path.name, "analysis_321.xlsx")
            self.assertEqual(result["extension"], "xlsx")
            self.assertEqual(result["writer"], "excel")
            self.assertTrue(path.exists())

    def test_save_analysis_dataframe_writes_txt_as_tsv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            df = pd.DataFrame({"name": ["alpha"], "amount": [10]})

            result = save_analysis_dataframe(df, temp_dir, "txt", timestamp=456)

            path = Path(result["file_path"])
            self.assertEqual(path.name, "analysis_456.txt")
            self.assertEqual(result["writer"], "tsv")
            self.assertIn("name\tamount", path.read_text(encoding="utf-8"))

    def test_save_analysis_dataframe_preserves_pdf_fallback_behavior(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            df = pd.DataFrame({"name": ["alpha"]})

            result = save_analysis_dataframe(df, temp_dir, "pdf", timestamp=789)

            path = Path(result["file_path"])
            self.assertEqual(path.name, "analysis_789.pdf")
            self.assertEqual(result["writer"], "csv_pdf_fallback")
            self.assertIn("alpha", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
