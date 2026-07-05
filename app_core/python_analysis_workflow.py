"""AI workflow for generating, validating, repairing, and running pandas code."""

from __future__ import annotations

import os

from app_core.analysis_utils import (
    clean_generated_python_code,
    detect_unsafe_python_keyword,
    validate_generated_python_code,
)
from app_core.python_analysis_runner import PythonAnalysisRunner
from app_core.utils import app_data_path


class PythonAnalysisWorkflowRunner:
    """Coordinate model-generated pandas analysis with validation-backed retries."""

    def __init__(self, bridge, max_attempts: int = 3):
        self.bridge = bridge
        self.max_attempts = max(1, int(max_attempts))

    def _build_prompt(self, dataframes: dict, instructions: str) -> str:
        schema_desc = self.bridge._create_enhanced_schema(dataframes)
        exact_columns = {table_name: list(df.columns) for table_name, df in dataframes.items()}
        instr_short = instructions[:6000] if len(instructions) > 6000 else instructions
        dataframe_names = ", ".join(list(dataframes.keys()))

        column_list = "EXACT COLUMN NAMES (use exactly as shown, do not correct spelling):\n"
        for table_name, cols in exact_columns.items():
            col_str = ", ".join([f"'{c}'" for c in cols])
            column_list += f"  {table_name}: [{col_str}]\n"

        multi_file_hint = ""
        if len(dataframes) > 1:
            multi_file_hint = (
                "MULTI-FILE RULE:\n"
                "- More than one dataframe is provided.\n"
                "- You MUST use ALL dataframes.\n"
                "- Do NOT ignore any dataframe.\n"
                "- Do NOT overwrite result_df multiple times.\n"
                "- Follow the USER TASK steps exactly to decide how to join/merge/compare.\n"
                "- Handle NULLs, duplicates, and type mismatches as needed."
            )

        return f"""
You are an expert Python data analyst.

TASK:
{instr_short}

IMPORTANT CONTEXT:
- Data is ALREADY LOADED into pandas DataFrames.
- DO NOT use pd.read_csv().
- DO NOT use file paths.
- DO NOT redefine dataframes.
- Use ONLY the existing DataFrames directly.

AVAILABLE DATAFRAMES:
{dataframe_names}

{column_list}

SCHEMA (columns + sample values):
{schema_desc}

{multi_file_hint}

STEP-INTERPRETATION PRIORITY:
- Follow the user's Steps section literally when present.
- If steps ask for unique_key, create unique_key.
- If steps say "all columns", derive columns dynamically from dataframe columns.
- Do not hardcode guessed column names when steps require all columns.
- If steps ask to delete/remove, perform those operations before final result.
- If steps ask arithmetic comparisons, compute numeric deltas/ratios/tolerance checks.
- If steps ask exact/fuzzy filtering (single or list), apply those filters explicitly.

STRICT RULES:
- Return ONLY valid Python code.
- Do NOT include explanations.
- Do NOT include markdown.
- Do NOT include comments.
- Do NOT include numbered steps.
- Do NOT include any text before or after the code.
- Do NOT import anything.
- Do NOT use read_csv.
- Do NOT use print().
- Do NOT create fake/sample data.
- Do NOT redefine dataframe variables.
- Assign final output ONLY ONCE to: result_df
- result_df must be a pandas DataFrame.
- Use column names EXACTLY as provided.

CODE REQUIREMENTS:
- Use only the provided DataFrames.
- Handle missing columns safely.
- If a column may not exist, check with: if 'col' in df.columns:
- For unique_key/all-columns reconciliation, generate dynamic column logic from df.columns.
- When comparing arithmetic across files, include computed difference columns in result_df when relevant.
- Ensure the code is executable as-is.

Return ONLY clean Python code.
"""

    @staticmethod
    def _repair_prompt(base_prompt: str, errors: list[dict]) -> str:
        fixes = []
        for item in errors[-3:]:
            error = str(item.get("error", "")).strip()
            code = str(item.get("code", "")).strip()
            if error:
                fixes.append(f"- {error}")
            if code:
                fixes.append(f"Previous code snippet:\n{code[:600]}")
        fix_block = "\n".join(fixes) if fixes else "- Fix the previous invalid code."
        return (
            base_prompt
            + "\n\nPREVIOUS ATTEMPT FAILED.\n"
            + "Apply ALL fixes below and return one corrected pandas code block only:\n"
            + fix_block
        )

    def _complete(self, prompt: str, attempt: int) -> str:
        response = self.bridge.model.create_completion(
            prompt,
            max_tokens=512,
            temperature=min(0.1 + attempt * 0.05, 0.25),
        )
        return response.get("choices", [{}])[0].get("text", "").strip()

    def execute(self, dataframes: dict, instructions: str, output_format: str, pipeline_status_messages: list = None) -> dict:
        try:
            status_messages = pipeline_status_messages if pipeline_status_messages else []

            def log(msg: str) -> None:
                status_messages.append(msg)
                print(msg)

            log("[PYTHON] Executing Python-based analysis...")
            log("[PYTHON] Building schema for AI...")

            if not self.bridge.model:
                return {"error": "No model loaded"}

            base_prompt = self._build_prompt(dataframes, instructions)
            errors: list[dict] = []
            output_dir = os.path.join(app_data_path(), "processed_files")
            runner = PythonAnalysisRunner(output_dir=output_dir, log=log)

            for attempt in range(self.max_attempts):
                log("[PYTHON] Generating Python code..." if attempt == 0 else f"[PYTHON] Repairing Python code (attempt {attempt + 1})...")
                prompt = base_prompt if attempt == 0 else self._repair_prompt(base_prompt, errors)
                raw_code = self._complete(prompt, attempt)
                log(f"[DEBUG] Raw LLM output:\n{raw_code[:500]}")

                python_code = clean_generated_python_code(raw_code)
                log("[PYTHON] Code generated successfully.")
                log(f"[PYTHON] Code preview:\n{python_code[:300]}")

                is_valid_code, validation_error = validate_generated_python_code(python_code)
                if not is_valid_code:
                    errors.append({"attempt": attempt + 1, "code": python_code, "error": validation_error})
                    log(f"[PYTHON] Validation failed: {validation_error}")
                    continue

                unsafe_keyword = detect_unsafe_python_keyword(python_code)
                if unsafe_keyword:
                    error = f"Unsafe code detected: {unsafe_keyword}"
                    errors.append({"attempt": attempt + 1, "code": python_code, "error": error})
                    log(f"[PYTHON] Validation failed: {error}")
                    continue

                print("\n" + "=" * 80)
                print("[PYTHON] FULL CODE TO BE EXECUTED:")
                print("=" * 80)
                print(python_code)
                print("=" * 80 + "\n")
                log("\n" + "=" * 80)
                log("[PYTHON] FULL CODE TO BE EXECUTED:")
                log("=" * 80)
                log(python_code)
                log("=" * 80 + "\n")

                runner_result = runner.run(
                    python_code=python_code,
                    dataframes=dataframes,
                    output_format=output_format,
                    status_messages=status_messages,
                )
                if runner_result.get("ok"):
                    if errors:
                        runner_result["repair_attempts"] = len(errors)
                    return runner_result

                error = runner_result.get("exception") or runner_result.get("error", "Unknown execution error")
                errors.append({"attempt": attempt + 1, "code": python_code, "error": error})
                log(f"[PYTHON] Attempt {attempt + 1} failed: {error}")

            last_error = errors[-1]["error"] if errors else "Unknown"
            return {
                "error": f"Python analysis failed after {len(errors)} attempt(s): {last_error}",
                "attempt_errors": errors,
            }
        except Exception as e:
            import traceback
            print(f"[PYTHON] Error: {e}")
            traceback.print_exc()
            return {"error": f"Python analysis failed: {str(e)}"}
