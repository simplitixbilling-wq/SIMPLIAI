import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app_core.file_output_writer import output_extension, save_analysis_dataframe, write_dataframe_to_path


class FileOutputWriterTests(unittest.TestCase):
    def test_output_extension_maps_formats(self):
        self.assertEqual(output_extension("excel"), "xlsx")
        self.assertEqual(output_extension("csv_json"), "csv")
        self.assertEqual(output_extension("bad", default="txt"), "txt")

    def test_save_analysis_dataframe_supports_custom_stem(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            df = pd.DataFrame({"a": [1]})

            result = save_analysis_dataframe(df, temp_dir, "csv", timestamp=111, stem="analysis_111_q2")

            path = Path(result["file_path"])
            self.assertEqual(path.name, "analysis_111_q2.csv")
            self.assertEqual(result["writer"], "csv")
            self.assertTrue(path.exists())

    def test_write_dataframe_to_path_uses_tsv_for_txt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            df = pd.DataFrame({"a": [1], "b": [2]})
            path = Path(temp_dir) / "out.txt"

            writer = write_dataframe_to_path(df, str(path), "txt")

            self.assertEqual(writer, "tsv")
            self.assertIn("a\tb", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
