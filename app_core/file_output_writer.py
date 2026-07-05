"""Shared DataFrame output writing helpers."""

from __future__ import annotations

import os
import time


OUTPUT_EXTENSIONS = {
    "excel": "xlsx",
    "xlsx": "xlsx",
    "csv": "csv",
    "csv_json": "csv",
    "pdf": "pdf",
    "txt": "txt",
}


def output_extension(output_format: str, default: str = "xlsx") -> str:
    fmt = str(output_format or "").strip().lower()
    return OUTPUT_EXTENSIONS.get(fmt, default)


def write_dataframe_to_path(result_df, output_file: str, extension: str) -> str:
    ext = str(extension or "").strip().lower()
    if ext == "xlsx":
        result_df.to_excel(output_file, index=False)
        return "excel"
    if ext == "csv":
        result_df.to_csv(output_file, index=False)
        return "csv"
    if ext == "pdf":
        result_df.to_csv(output_file, index=False)
        return "csv_pdf_fallback"
    result_df.to_csv(output_file, sep="\t", index=False)
    return "tsv"


def save_analysis_dataframe(
    result_df,
    output_dir: str,
    output_format: str,
    timestamp: int | None = None,
    stem: str | None = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    ts = int(time.time()) if timestamp is None else int(timestamp)
    ext = output_extension(output_format, default="xlsx")
    filename_stem = stem or f"analysis_{ts}"
    output_file = os.path.join(output_dir, f"{filename_stem}.{ext}")
    writer = write_dataframe_to_path(result_df, output_file, ext)
    return {
        "file_path": output_file,
        "extension": ext,
        "timestamp": ts,
        "writer": writer,
    }
