"""Execution runner for generated pandas analysis code."""

from __future__ import annotations

import os
import time
import traceback
from typing import Callable

import pandas as pd

from app_core.analysis_utils import (
    clean_analysis_summary,
    detect_unsafe_python_keyword,
    validate_generated_python_code,
)
from app_core.file_output_writer import output_extension, save_analysis_dataframe


class PythonAnalysisRunner:
    """Run validated generated Python against pre-loaded pandas DataFrames."""

    def __init__(
        self,
        output_dir: str,
        log: Callable[[str], None] | None = None,
        time_fn: Callable[[], float] | None = None,
    ):
        self.output_dir = output_dir
        self.log = log or (lambda _msg: None)
        self.time_fn = time_fn or time.time

    def run(
        self,
        python_code: str,
        dataframes: dict,
        output_format: str,
        status_messages: list,
    ) -> dict:
        is_valid_code, validation_error = validate_generated_python_code(python_code)
        if not is_valid_code:
            return {"error": validation_error}

        unsafe_keyword = detect_unsafe_python_keyword(python_code)
        if unsafe_keyword:
            return {"error": f"Unsafe code detected: {unsafe_keyword}"}

        allowed_builtins = {"len": len, "range": range}
        namespace = {
            "__builtins__": allowed_builtins,
            "pd": pd,
            "result_df": None,
        }
        namespace.update(dataframes)

        self.log("[DEBUG] Namespace keys before exec:")
        for key in list(namespace.keys())[:10]:
            self.log(f"  - {key}: {type(namespace.get(key)).__name__}")

        self.log("[PYTHON] Running generated code...")
        start_time = self.time_fn()
        self.log(f"[DEBUG] Start time: {start_time}")

        try:
            exec(python_code, namespace)
        except Exception as exc:
            error_trace = traceback.format_exc()
            self.log(f"[PYTHON] Execution failed: {str(exc)}")
            self.log(f"[DEBUG] Full traceback:\n{error_trace}")
            return {
                "error": f"Python execution failed: {str(exc)}",
                "exception": str(exc),
                "traceback": error_trace,
            }

        execution_time = round(self.time_fn() - start_time, 2)
        self.log(f"[DEBUG] Exec completed after {execution_time}s")
        self.log(f"[PYTHON] Code executed successfully in {execution_time}s")

        self.log("[DEBUG] Checking namespace after execution...")
        if "result_df" in namespace:
            result_obj = namespace["result_df"]
            self.log(f"[DEBUG] result_df exists: {type(result_obj).__name__}")
        else:
            self.log("[DEBUG] result_df NOT in namespace (BAD)")

        self.log("[PYTHON] Checking result_df...")
        result_df = namespace.get("result_df")
        if result_df is None:
            self.log("[DEBUG] ERROR: result_df is None")
            self.log(f"[DEBUG] Available keys in namespace: {list(namespace.keys())}")
            return {"error": "No result_df produced"}

        if not isinstance(result_df, pd.DataFrame):
            self.log(f"[DEBUG] ERROR: result_df is {type(result_df).__name__}, not DataFrame")
            return {"error": f"result_df is not a DataFrame (got {type(result_df).__name__})"}

        row_count = len(result_df)
        col_count = len(result_df.columns)
        self.log(f"[DEBUG] Row count: {row_count}, Column count: {col_count}")
        self.log(f"[PYTHON] Valid DataFrame with {row_count} rows, {col_count} columns")
        self.log(f"[PYTHON] Execution completed in {execution_time}s")
        self.log("[PYTHON] Saving results to file...")

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            timestamp = int(self.time_fn())
            ext = output_extension(output_format, default="xlsx")
            self.log(f"[DEBUG] Output format: {output_format} -> extension: {ext}")
            save_result = save_analysis_dataframe(
                result_df, self.output_dir, output_format, timestamp=timestamp
            )
            output_file = save_result["file_path"]
            self.log(f"[DEBUG] Output file path: {output_file}")
            self.log(f"[DEBUG] DataFrame writer: {save_result['writer']}")
            self.log(f"[DEBUG] Verifying file exists: {os.path.exists(output_file)}")
            self.log(f"[PYTHON] Saved results to: {output_file}")
        except Exception as exc:
            error_trace = traceback.format_exc()
            self.log(f"[PYTHON] Error saving file: {exc}")
            self.log(f"[DEBUG] Save error traceback:\n{error_trace}")
            return {"error": f"Python save failed: {str(exc)}"}

        summary = clean_analysis_summary(status_messages, f"Total rows: {row_count}")
        return {
            "ok": True,
            "response_text": summary,
            "file_path": output_file,
            "row_count": row_count,
            "col_count": col_count,
            "execution_time": execution_time,
        }
