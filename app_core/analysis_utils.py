"""Pure helpers for file-analysis and generated-code workflows."""

from __future__ import annotations

import ast
import re
from app_core.file_output_writer import output_extension, save_analysis_dataframe


PYTHON_BANNED_FRAGMENTS = (
    "read_csv",
    "path_to_",
    "import pandas",
    "import os",
    "import sys",
    "import subprocess",
    "exec(",
    "eval(",
    "__import__",
    "open(",
    "os.",
    "system(",
    "popen(",
    "print(",
)

PYTHON_UNSAFE_KEYWORDS = (
    "__import__",
    "exec",
    "eval",
    "open",
    "system",
    "popen",
    "os.",
)

PYTHON_BANNED_AST_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
}

PYTHON_BANNED_AST_NODES = (
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Delete,
    ast.FunctionDef,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.With,
)

DEBUG_SUMMARY_TAGS = (
    "[SQL-DEBUG]",
    "[CODE_EXEC]",
    "[DEBUG]",
    "clean_sql input",
    "clean_sql output",
    "clean_sql:",
    "VERIFY COLUMN",
    "VERIFY TABLE",
    "repair_sql",
    "semantic_review",
)

def clean_generated_python_code(code: str) -> str:
    """Remove common markdown/explanation noise around model-generated Python."""
    lines = str(code or "").split("\n")
    clean_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("```"):
            continue
        if stripped[0].isdigit() and "." in stripped[:3]:
            continue
        if stripped.lower().startswith(("here", "to ", "this ", "step")):
            continue
        clean_lines.append(line)

    return "\n".join(clean_lines)


def validate_generated_python_code(code: str) -> tuple[bool, str]:
    """Validate generated analysis code before executing it in Bridge."""
    candidate = str(code or "").strip()
    if not candidate:
        return False, "Invalid code: empty"

    lowered = candidate.lower()
    for fragment in PYTHON_BANNED_FRAGMENTS:
        if fragment.lower() in lowered:
            return False, f"Invalid code: banned fragment '{fragment}'"

    if "result_df" not in candidate:
        return False, "Invalid code: result_df not found"

    try:
        tree = ast.parse(candidate, filename="<generated_code>", mode="exec")
    except SyntaxError as exc:
        return False, f"Invalid code: syntax error: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, PYTHON_BANNED_AST_NODES):
            return False, f"Invalid code: banned syntax '{type(node).__name__}'"
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in PYTHON_BANNED_AST_CALLS:
                return False, f"Invalid code: banned name '{node.id}'"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return False, f"Invalid code: banned attribute '{node.attr}'"
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name) and target.id in PYTHON_BANNED_AST_CALLS:
                return False, f"Invalid code: banned call '{target.id}'"

    compile(tree, "<generated_code>", "exec")
    return True, ""


def detect_unsafe_python_keyword(code: str) -> str:
    """Return the first unsafe execution keyword found, or an empty string."""
    candidate = str(code or "")
    for keyword in PYTHON_UNSAFE_KEYWORDS:
        if keyword in candidate:
            return keyword
    return ""


def clean_analysis_summary(messages: list, suffix: str = "") -> str:
    """Drop debug noise from pipeline status messages and append an optional suffix."""
    clean = []
    for msg in messages or []:
        line = str(msg or "")
        if any(tag in line for tag in DEBUG_SUMMARY_TAGS):
            continue
        line = re.sub(r"^\[(SQL|PYTHON|AGENT)\]\s*", "", line).strip()
        if line:
            clean.append(line)

    result = "\n".join(clean) if clean else "Analysis completed."
    if suffix:
        result += "\n\n" + str(suffix)
    return result
