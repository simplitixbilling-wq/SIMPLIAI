"""SQL analysis runner extracted from the pywebview bridge.

The runner owns SQL generation, validation, retry, execution, and export orchestration.
It delegates app-specific services such as model access, schema generation, and stop flags
back to the Bridge instance passed at construction time.
"""

from __future__ import annotations

import json
import os
import re
import time

import pandas as pd

from app_core.analysis_utils import output_extension, save_analysis_dataframe
from app_core.sql_analysis_utils import (
    apply_fuzzy_table_mapping,
    clean_sql_text,
    extract_direct_sql,
    repair_missing_table_errors,
    repair_sql_identifiers as repair_sql_identifiers_text,
    repair_sql_join_predicates as repair_sql_join_predicates_text,
    split_sql_statements,
    strip_non_sql_prefix,
    validate_sql_basic_structure,
    validate_sql_table_usage,
)
from app_core.utils import app_data_path


_RE_THINK = re.compile(r"<think>.*?</think>|<\|channel>thought.*?<channel\|>", re.DOTALL)
_RE_THINK_INCOMPLETE = re.compile(r"<think>.*$|<\|channel>thought.*$", re.DOTALL)


class SQLAnalysisRunner:
    def __init__(self, bridge):
        self.bridge = bridge

    def execute(self, dataframes: dict, instructions: str, output_format: str, pipeline_status_messages: list = None, context_block: str = "") -> dict:
        """Generate and execute SQL analysis using DuckDB."""
        bridge = self.bridge
        try:
            import duckdb
            import pandas as pd

            status_messages = pipeline_status_messages if pipeline_status_messages else []
            def _debug_status(text: str):
                status_messages.append(text)
                print(text)

            def _clean_summary(msgs: list, suffix: str = "") -> str:
                """Build a user-friendly summary from status_messages.
                
                Strips debug/internal tags and keeps only meaningful lines.
                """
                clean = []
                for m in msgs:
                    # Skip internal debug lines entirely
                    if any(tag in m for tag in (
                        "[SQL-DEBUG]", "[CODE_EXEC]", "[DEBUG]",
                        "clean_sql input", "clean_sql output",
                        "clean_sql:", "VERIFY COLUMN", "VERIFY TABLE",
                        "repair_sql", "semantic_review",
                    )):
                        continue
                    # Strip tag prefixes for user-facing lines
                    line = re.sub(r'^\[(SQL|PYTHON|AGENT)\]\s*', '', m).strip()
                    if line:
                        clean.append(line)
                result = "\n".join(clean) if clean else "Analysis completed."
                if suffix:
                    result += "\n\n" + suffix
                return result

            def _stopped_result() -> dict:
                _debug_status("[SQL] Stopped by user")
                return {"error": "Generation stopped by user", "stopped": True}

            msg = "[SQL] Executing SQL-based analysis..."
            status_messages.append(msg)
            print(msg)

            if bridge.stop_generation_flag:
                return _stopped_result()

            # Final safety normalization: enforce SQL-safe column names for every dataframe
            # right before DuckDB registration (covers any loader path).
            normalized_dataframes = {}
            for table_name, df in dataframes.items():
                working = df.copy()
                old_cols = [str(c) for c in working.columns]
                safe_cols = bridge._normalize_column_names(old_cols)
                if old_cols != safe_cols:
                    working.columns = safe_cols
                    preview = ", ".join(
                        f"{o}->{n}" for o, n in list(zip(old_cols, safe_cols))[:8] if o != n
                    )
                    if preview:
                        _debug_status(f"[SQL] Normalized columns in {table_name}: {preview}")
                normalized_dataframes[table_name] = working
            dataframes = normalized_dataframes
            
            # Create DuckDB connection
            conn = duckdb.connect(":memory:")
            
            # Register dataframes as tables
            for table_name, df in dataframes.items():
                conn.register(table_name, df)
                msg = f"[SQL] Registered table: {table_name}"
                status_messages.append(msg)
                print(msg)
            
            # Generate SQL code via AI - with FULL schema so AI understands all columns
            msg = "[SQL] Building complete schema for AI code generation..."
            status_messages.append(msg)
            print(msg)
            # Build schema using actual DuckDB column types (not pandas dtypes)
            # so the model knows which columns are numeric vs text
            schema_desc = bridge._create_duckdb_schema(conn, dataframes)
            
            # Extract exact column names for explicit instruction to AI
            exact_columns = {}
            for table_name, df in dataframes.items():
                exact_columns[table_name] = list(df.columns)
            
            # Build column list with DuckDB types (accurate types after registration)
            column_list = "Table columns (name: DuckDB_type):\n"
            for table_name in exact_columns:
                column_list += f"  {table_name}:\n"
                try:
                    describe_rows = conn.execute(f"DESCRIBE {table_name}").fetchall()
                    for row in describe_rows:
                        col_name, col_type = row[0], row[1]
                        column_list += f"    - {col_name}: {col_type}\n"
                except Exception:
                    for col in exact_columns[table_name]:
                        column_list += f"    - {col}\n"
            
            # Minimal flags for output-format decisions only (not for SQL generation)
            instr_lower = instructions.lower()
            is_reconciliation = any(kw in instr_lower for kw in ["reconcil", "mismatch", "difference", "compare", "find diff"])

            # Keep full steps context for SQL generation
            instr_short = instructions[:6000] if len(instructions) > 6000 else instructions

            # Build a mapping hint so model understands user's file aliases
            # e.g. "Report_A", "first file", "file 1"  -> actual DuckDB table name
            table_names_list = list(dataframes.keys())


            def _apply_fuzzy_table_mapping(sql_text: str) -> tuple[str, dict]:
                """Map table tokens in FROM/JOIN to loaded DuckDB table names (exact -> normalized -> fuzzy)."""
                return apply_fuzzy_table_mapping(sql_text, dataframes.keys())

            def _repair_missing_table_errors(sql_text: str, error_text: str) -> tuple[str, dict]:
                """When DuckDB reports missing table names, try alias/fuzzy remap and return updated SQL."""
                return repair_missing_table_errors(sql_text, error_text, dataframes.keys())
            mapping_lines = []
            ordinals = ["first", "second", "third", "fourth"]
            for i, tname in enumerate(table_names_list):
                # original filename (strip df_ prefix and replace _ with space)
                friendly = tname[3:].replace('_', ' ') if tname.startswith('df_') else tname
                ordinal = ordinals[i] if i < len(ordinals) else str(i + 1)
                mapping_lines.append(
                    f"  '{tname}' - {ordinal} file, also known as: \"{friendly}\", "
                    f"\"Report_{'ABCDEFGH'[i]}\", \"{ordinal} file\", \"file {i+1}\""
                )
            table_alias_hint = "TABLE ALIASES (user may refer to tables by these names):\n" + "\n".join(mapping_lines) + "\n"

            context_section = f"\n\nADDITIONAL CONTEXT FILES (read-only reference, not in SQL tables):\n{context_block}" if context_block else ""

            sql_prompt = f"""Generate a single DuckDB SQL query for the task below.

USER TASK:
{instr_short}{context_section}

{table_alias_hint}
AVAILABLE TABLES AND COLUMN TYPES:
{column_list}
RULES:
- Follow the USER TASK steps exactly. The user's instructions are the highest priority.
- Output ONLY the SQL query. No explanation, no code fence, no comments.
- Use column names EXACTLY as listed above (do not correct spelling).
- If a column name contains spaces/special chars or starts with a digit, reference it with double quotes (e.g., T."Vendor name", T."2A Value").
- Table aliases must start with a letter or underscore (do NOT use aliases like 2A). Prefer aliases like A, B, T1, T2.
- If user says ignore/exclude/remove a column, that column must NOT appear from the Steps mentioned.
- If user asks to rename a column, return only the renamed alias from the Steps mentioned.
- If user asks aggregation/grouping, query MUST include corresponding GROUP BY and aggregate functions.
- Use CAST(col AS DOUBLE) or TRY_CAST(col AS DOUBLE) before aggregating numeric columns when the type is VARCHAR.
- For key comparisons/joins where NULLs may appear, prefer IS NOT DISTINCT FROM or COALESCE normalization.
- For text filters, account for NULL safely (e.g., COALESCE(col, '') before LIKE/ILIKE when needed).
- Handle NULLs explicitly in join keys/comparisons.
- if its a multi step request use WITH statement only else Select Statement only.
- Do NOT add WHERE filters on columns that the user did not ask to filter. Only filter on conditions explicitly stated in the task.
- If a computation applies conditionally (e.g. 12% of Basic where PF=Yes), use CASE WHEN in the SELECT, not a WHERE clause that removes rows.
- CRITICAL: Only reference columns that are explicitly listed in AVAILABLE TABLES AND COLUMN TYPES above. NEVER write a column name that does not appear in that list (e.g. do not write A.gross if "gross" is not listed - instead write the arithmetic expression: A.basic + A.hra + A.conveyance + A.special_allowance AS gross).
- CRITICAL: Do NOT reference SELECT-level aliases inside the same SELECT or in WHERE/expressions. DuckDB does not allow this. Re-compute the expression inline or use a WITH clause."""


            def repair_sql_identifiers(sql_text: str) -> str:
                table_columns = {name: list(df.columns) for name, df in dataframes.items()}
                return repair_sql_identifiers_text(sql_text, table_columns, debug=_debug_status)

            def repair_sql_join_predicates(sql_text: str) -> str:
                return repair_sql_join_predicates_text(sql_text, debug=_debug_status)

            def semantic_review_sql(task_text: str, sql_text: str) -> tuple[bool, str]:
                """Dynamically validate SQL against user task using the model itbridge.

                Returns (ok, reason). If reviewer fails, default to pass to avoid blocking.
                """
                if not bridge.model:
                    return True, ""

                # -- Keyword pre-screen (no model call needed) --------------------------
                # Extract numbers and bare words from the task, check they appear in SQL.
                # This catches obvious cases where a small model wrongly flags a valid SQL.
                sql_lower = sql_text.lower()
                task_lower = task_text.lower()
                # Collect all numbers mentioned in the task (e.g. 0.12, 12%, 200)
                task_numbers = re.findall(r'\d+\.?\d*', task_lower)
                # Collect key column-like words from the task (>= 4 chars, not SQL keywords)
                _SQL_KW = {'from', 'join', 'where', 'group', 'order', 'having', 'select',
                           'with', 'case', 'when', 'then', 'else', 'end', 'and', 'not',
                           'null', 'like', 'cast', 'coalesce', 'apply', 'each', 'employee',
                           'output', 'compute', 'calculate', 'generate', 'produce', 'using'}
                task_words = [w for w in re.findall(r'[a-z_]{4,}', task_lower)
                              if w not in _SQL_KW]
                # Check: at least 80% of task numbers appear in SQL (as substrings)
                nums_found = sum(1 for n in task_numbers if n in sql_lower)
                nums_ok = (not task_numbers) or (nums_found / len(task_numbers) >= 0.8)
                # Check: at least 60% of task keywords appear in SQL
                words_found = sum(1 for w in task_words if w in sql_lower)
                words_ok = (not task_words) or (words_found / len(task_words) >= 0.6)
                if nums_ok and words_ok:
                    _debug_status(f"[SQL-DEBUG] semantic_review_sql: keyword pre-screen PASS "
                                  f"(nums {nums_found}/{len(task_numbers)}, "
                                  f"words {words_found}/{len(task_words)}) - skipping model call")
                    return True, ""
                # -- End pre-screen -----------------------------------------------------
                review_user = f"""Review whether the SQL fully satisfies the USER TASK.

USER TASK:
{task_text}

SQL TO REVIEW:
{sql_text}

Return ONLY strict JSON with keys:
- ok: true or false
- reason: short string
- missing_requirements: array of short strings

Rules:
- Mark ok=false if SQL misses any explicit task step.
- Mark ok=false if SQL uses only one table when task requires reconciling two files.
- Mark ok=false if SQL compares only missing rows but task asks value comparisons on matched rows.
- Mark ok=false if task asks summarize/aggregate/group and SQL does not aggregate accordingly.
- Mark ok=false if user says ignore/exclude/remove a column but SQL still includes it in final output.
- Mark ok=false if user asks renaming but SQL returns old column name instead of requested alias.
"""
                try:
                    review_prompt = bridge._build_chat_prompt(
                        system="You are a strict SQL task compliance checker. Return JSON only.",
                        messages=[],
                        user_text=review_user,
                        extra_context="",
                    )
                    with bridge.model_lock:
                        review_resp = bridge.model.create_completion(
                            review_prompt,
                            max_tokens=400,
                            temperature=0.0,
                            stop=bridge._get_stop_tokens(),
                        )
                    raw = review_resp.get("choices", [{}])[0].get("text", "").strip()
                    raw = _RE_THINK.sub("", raw).strip()
                    raw = _RE_THINK_INCOMPLETE.sub("", raw).strip()
                    fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
                    if fence_match:
                        raw = fence_match.group(1).strip()
                    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
                    if json_match:
                        raw = json_match.group(0).strip()
                    payload = json.loads(raw)
                    ok = bool(payload.get("ok", True))
                    reason = str(payload.get("reason", "")).strip()
                    missing = payload.get("missing_requirements", [])
                    if isinstance(missing, list) and missing:
                        # Keep full missing requirements list; truncation loses actionable constraints.
                        reason = (reason + " | Missing: " + "; ".join(str(x) for x in missing)).strip(" |")
                    return ok, reason
                except Exception as e:
                    _debug_status(f"[SQL-DEBUG] semantic_review_sql skipped due to reviewer error: {e}")
                    return True, ""


            def validate_sql(sql_text: str, semantic: bool = True) -> tuple[str | None, str]:
                """Validate SQL in cost-ordered phases: structural  -> EXPLAIN  -> tables  -> semantic."""
                _debug_status(f"[SQL-DEBUG] validate_sql input (len={len(sql_text) if sql_text else 0}): {(sql_text or '')[:200]!r}")
                if not sql_text or not sql_text.strip():
                    _debug_status("[SQL-DEBUG] validate_sql: FAIL - empty input")
                    return "Empty SQL generated", ""

                normalized = strip_non_sql_prefix(sql_text.strip())

                statements = split_sql_statements(normalized)
                if not statements:
                    return "Empty SQL generated", ""
                if len(statements) > 1:
                    _debug_status("[SQL-DEBUG] validate_sql: FAIL - multiple SQL statements detected")
                    return "Multiple SQL statements detected; expected exactly one", normalized
                normalized = statements[0]
                if sql_text.strip().endswith(";"):
                    _debug_status(f"[SQL-DEBUG] validate_sql: stripped trailing semicolon")

                # -- Phase 1: Deterministic structural checks (free) --
                structural_error = validate_sql_basic_structure(normalized, dataframes.keys())
                if structural_error:
                    _debug_status(f"[SQL-DEBUG] validate_sql: FAIL - {structural_error}")
                    return structural_error, normalized

                # -- Phase 2: DuckDB EXPLAIN plan (catches syntax errors before model calls) --
                try:
                    conn.execute(f"EXPLAIN {normalized}").fetchall()
                    _debug_status(f"[SQL-DEBUG] validate_sql: EXPLAIN plan OK")
                except Exception as e:
                    _debug_status(f"[SQL-DEBUG] validate_sql: FAIL - EXPLAIN plan error: {e}")
                    return f"SQL parse/plan failed: {e}", normalized

                # -- Phase 3: Table reference checks (free) --
                table_usage_error = validate_sql_table_usage(normalized, dataframes.keys(), is_reconciliation=is_reconciliation)
                if table_usage_error:
                    return table_usage_error, normalized

                # -- Phase 4: Semantic review via model (optional) --
                if semantic:
                    sem_ok, sem_reason = semantic_review_sql(instr_short, normalized)
                    if not sem_ok:
                        # Add machine-readable payload for repair prompt consumption.
                        missing_items = []
                        m = re.search(r"\|\s*Missing:\s*(.+)$", sem_reason or "", re.IGNORECASE)
                        if m:
                            missing_items = [x.strip().rstrip(".") for x in m.group(1).split(";") if x.strip()]
                        missing_json = json.dumps(missing_items, ensure_ascii=True)
                        return f"SQL does not satisfy task: {sem_reason or 'semantic mismatch'} | MISSING_REQ_JSON: {missing_json}", normalized

                _debug_status(f"[SQL-DEBUG] validate_sql: PASS - SQL looks valid")
                return None, normalized

            # -- Direct SQL path: if user already supplied SQL, skip AI generation --
            direct_sql = extract_direct_sql(instructions)
            if direct_sql:
                _debug_status("[SQL] Direct SQL detected in instructions; skipping AI SQL generation")
                sql_code = clean_sql_text(direct_sql, debug=_debug_status)
                sql_code, table_replacements = _apply_fuzzy_table_mapping(sql_code)
                if table_replacements:
                    _debug_status(f"[SQL] Fuzzy table mapping applied: {table_replacements}")
                sql_code = repair_sql_identifiers(sql_code)
                sql_code = repair_sql_join_predicates(sql_code)

                direct_statements = split_sql_statements(sql_code)
                if len(direct_statements) > 1:
                    _debug_status(f"[SQL] Detected {len(direct_statements)} direct SQL statements; executing each query separately")
                    output_dir = os.path.join(app_data_path(), "processed_files")
                    os.makedirs(output_dir, exist_ok=True)
                    timestamp = int(time.time())
                    saved_paths = []
                    total_rows = 0

                    for idx, statement in enumerate(direct_statements, start=1):
                        if bridge.stop_generation_flag:
                            conn.close()
                            return _stopped_result()

                        statement = strip_non_sql_prefix(statement)

                        validation_error, statement = validate_sql(statement, semantic=False)
                        if validation_error:
                            conn.close()
                            return {"error": f"Direct SQL validation failed (Query {idx}): {validation_error}"}

                        result_df = conn.execute(statement).fetchdf()
                        row_count = len(result_df)
                        total_rows += row_count

                        save_result = save_analysis_dataframe(
                            result_df,
                            output_dir,
                            output_format,
                            timestamp=timestamp,
                            stem=f"analysis_{timestamp}_q{idx}",
                        )
                        output_file = save_result["file_path"]
                        saved_paths.append(output_file)
                        _debug_status(f"[SQL] Query {idx} executed successfully: {row_count} rows -> {output_file}")

                    summary = _clean_summary(status_messages, f"Total queries: {len(direct_statements)} | Total rows: {total_rows}")
                    conn.close()
                    return {
                        "ok": True,
                        "response_text": summary,
                        "file_path": saved_paths[0] if saved_paths else None,
                        "file_paths": saved_paths,
                    }

                validation_error, sql_code = validate_sql(sql_code, semantic=False)
                if validation_error:
                    if "Table with name" in validation_error and "does not exist" in validation_error:
                        repaired_sql, missing_replacements = _repair_missing_table_errors(sql_code, validation_error)
                        if missing_replacements:
                            _debug_status(f"[SQL] Missing-table recovery mapping applied: {missing_replacements}")
                            validation_error, sql_code = validate_sql(repaired_sql, semantic=False)
                    if validation_error:
                        return {"error": f"Direct SQL validation failed: {validation_error}"}

                _debug_status("[SQL] Direct SQL validation passed; executing query on DuckDB")
            else:
                sql_code = ""

            # -- Generate  -> Validate  -> Repair loop (retry until success or stuck) --
            MAX_SQL_ATTEMPTS = 15  # safety ceiling

            if not direct_sql:
                msg = "[SQL] Generating SQL query..."
                status_messages.append(msg)
                print(msg)
                if not bridge.model:
                    return {"error": "No model loaded"}

                # Print full sql_prompt so it can be verified in logs
                _debug_status(f"[SQL-DEBUG] Full sql_prompt ({len(sql_prompt)} chars):\n{'='*60}\n{sql_prompt}\n{'='*60}")

                stop_tokens = bridge._get_stop_tokens()
                errors_so_far = []  # tracks {"attempt": N, "sql": str, "error": str}
                _consecutive_same_error = 0  # detect when model is stuck
                validation_error = None
                retry_exhausted = False
                model_stuck = False

                for attempt in range(MAX_SQL_ATTEMPTS):
                    if bridge.stop_generation_flag:
                        return _stopped_result()
                    _debug_status(f"[SQL-DEBUG] -- Attempt {attempt + 1}/{MAX_SQL_ATTEMPTS} --")

                    if attempt == 0:
                        # First attempt: generate from original prompt
                        gen_prompt = bridge._build_chat_prompt(
                            system="You are a SQL expert. Output ONLY a single DuckDB SELECT query with no explanation.",
                            messages=[],
                            user_text=sql_prompt,
                            extra_context="",
                        )
                    else:
                        # Repair: include the failed SQL + specific fix instructions
                        prev = errors_so_far[-1]
                        # Extract actionable fixes from error messages
                        fix_items = []
                        for e in errors_so_far:
                            err = e["error"]
                            # Prefer machine-readable missing requirements if present.
                            json_match = re.search(r"MISSING_REQ_JSON:\s*(\[[\s\S]*\])", err)
                            if json_match:
                                try:
                                    parsed = json.loads(json_match.group(1))
                                    if isinstance(parsed, list):
                                        for item in parsed:
                                            item = str(item).strip().rstrip(".")
                                            if item:
                                                fix_items.append(f"- {item}")
                                        continue
                                except Exception:
                                    pass
                            # Pull out "Missing: ..." items from semantic reviewer
                            missing_match = re.search(r"Missing:\s*(.+)", err)
                            if missing_match:
                                for item in missing_match.group(1).split(";"):
                                    item = item.strip().rstrip(".")
                                    if item:
                                        fix_items.append(f"- {item}")
                            else:
                                fix_items.append(f"- Fix: {err}")

                        # Deduplicate fix items
                        fix_items = list(dict.fromkeys(fix_items))
                        fix_block = "\n".join(fix_items) if fix_items else f"- {prev['error']}"
                        # Truncate failed SQL to 600 chars - model only needs to see the
                        # structure that failed, not fill the entire context window with garbage.
                        prev_sql_snippet = prev['sql'][:600] + ("..." if len(prev['sql']) > 600 else "")
                        repair_user = (
                            sql_prompt
                            + f"\n\nYOUR PREVIOUS SQL (which failed validation):\n{prev_sql_snippet}"
                            + f"\n\nREQUIRED FIXES (apply ALL of these):\n{fix_block}"
                            + "\n\nRewrite the SQL to fix ALL issues above. Return exactly ONE read-only DuckDB SQL query starting with SELECT or WITH."
                        )
                        gen_prompt = bridge._build_chat_prompt(
                            system="You are a SQL expert. Output ONLY a single DuckDB SELECT query with no explanation.",
                            messages=[],
                            user_text=repair_user,
                            extra_context="",
                        )
                        _debug_status(f"[SQL-DEBUG] Repair prompt length: {len(gen_prompt)} chars")

                    try:
                        if bridge.stop_generation_flag:
                            return _stopped_result()
                        raw_sql = ""
                        with bridge.model_lock:
                            if bridge.model is None:
                                return {"error": "No model loaded"}

                            # Stream tokens so stop_generation can interrupt mid-attempt.
                            # Cap at 700 tokens - SQL queries don't need more, and higher limits
                            # let small models fill context with infinite nested SELECT loops.
                            stream = bridge.model(
                                gen_prompt,
                                max_tokens=700,
                                temperature=min(0.1 + (attempt * 0.08), 0.9),  # ramp up creativity on retries
                                stream=True,
                                stop=stop_tokens,
                            )

                            for chunk in stream:
                                if bridge.stop_generation_flag:
                                    return _stopped_result()
                                raw_sql += chunk.get("choices", [{}])[0].get("text", "")

                        if bridge.stop_generation_flag:
                            return _stopped_result()
                    except Exception as model_err:
                        _debug_status(f"[SQL-DEBUG] model.create_completion RAISED on attempt {attempt+1}: {model_err}")
                        import traceback; traceback.print_exc()
                        return {"error": f"Model inference failed: {model_err}"}

                    _debug_status(f"[SQL-DEBUG] Attempt {attempt+1} raw output (len={len(raw_sql)}):\n{'='*60}\n{raw_sql}\n{'='*60}")

                    raw_sql = raw_sql.strip()

                    # Detect "SELECT bomb": model stuck in infinite nested SELECT loop.
                    # Count FROM ( occurrences - more than 6 deep means garbage output.
                    if raw_sql.upper().count("FROM (") > 6:
                        _debug_status(f"[SQL-DEBUG] SELECT bomb detected ({raw_sql.upper().count('FROM (')} nesting levels) - discarding output")
                        errors_so_far.append({"attempt": attempt, "sql": "", "error": "Output contained infinitely nested SELECT subqueries - rewrite as a flat JOIN or simple WITH clause"})
                        continue

                    sql_code = clean_sql_text(raw_sql, debug=_debug_status)
                    sql_code = repair_sql_identifiers(sql_code)
                    sql_code = repair_sql_join_predicates(sql_code)

                    # -- Build real-column sets (global + per alias) --
                    all_real_cols = set()
                    for _df in dataframes.values():
                        all_real_cols.update(c.lower() for c in _df.columns)
                    # Map table alias  -> set of real column names for that table
                    _alias_to_cols: dict = {}
                    for _m in re.finditer(r'\b(df_\w+)\s+([A-Za-z_]\w*)\b', sql_code, re.IGNORECASE):
                        _tbl, _alias = _m.group(1), _m.group(2).upper()
                        for _df_name, _df in dataframes.items():
                            if _df_name.lower() == _tbl.lower():
                                _alias_to_cols[_alias] = {c.lower() for c in _df.columns}
                                break

                    def _col_ref_valid(alias: str, col: str) -> bool:
                        """Return True if alias.col is a valid reference."""
                        a = alias.upper()
                        c = col.lower()
                        if c in ('null', 'not', 'true', 'false'):
                            return True
                        if a in _alias_to_cols:
                            return c in _alias_to_cols[a]
                        return c in all_real_cols  # unqualified / unknown alias

                    # -- Strip WHERE conditions that reference non-existent columns --
                    # Also drops any surviving WHERE conditions when the task contains no
                    # explicit row-filtering language - making the fix generic, not
                    # tied to specific values like 'Yes'/'No' or IS NOT NULL patterns.
                    def _strip_bad_where_conditions(sql_text: str) -> str:
                        where_match = re.search(r'\bWHERE\b([\s\S]*?)(?=\b(?:GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|UNION|EXCEPT|INTERSECT)\b|$)', sql_text, re.IGNORECASE)
                        if not where_match:
                            return sql_text
                        where_body = where_match.group(1)
                        # If more closing parens than opening, this WHERE is inside a
                        # subquery and our regex has captured the closing ')' of that
                        # subquery plus JOIN clauses. Do not touch it - we'd corrupt the SQL.
                        if where_body.count(')') > where_body.count('('):
                            _debug_status("[SQL-DEBUG] WHERE spans subquery boundary - skipping sanitizer")
                            return sql_text
                        conditions = re.split(r'\bAND\b', where_body, flags=re.IGNORECASE)
                        good = []
                        for cond in conditions:
                            cond_s = cond.strip()
                            if not cond_s:
                                continue
                            # Strip conditions with non-existent alias.col refs (schema check)
                            aliased = re.findall(r'\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b', cond_s)
                            bad = [(a, c) for a, c in aliased if not _col_ref_valid(a, c)]
                            if bad:
                                _debug_status(f"[SQL-DEBUG] Stripped invalid WHERE condition "
                                              f"(bad refs: {bad}): {cond_s[:100]}")
                                continue
                            good.append(cond_s)
                        if not good:
                            return sql_text[:where_match.start()].rstrip() + sql_text[where_match.end():]
                        # If valid conditions survived, check whether the task actually
                        # asked for row filtering. Small models add defensive WHERE
                        # conditions that were never requested.
                        # Explicit filter intent: comparison with a literal value/number,
                        # or filter/exclude/only keywords in a filtering context.
                        _filter_intent = bool(re.search(
                            r'\b(?:filter|exclude|only\s+(?:include|show|rows?|records?))\b'
                            r'|(?:>|<|>=|<=|!=|<>)\s*[\d\'""]',
                            instructions, re.IGNORECASE))
                        if not _filter_intent:
                            _debug_status(f"[SQL-DEBUG] Task has no row-filtering intent - "
                                          f"dropped {len(good)} surviving WHERE condition(s)")
                            # Use a space separator to avoid joining last token with next keyword
                            return sql_text[:where_match.start()].rstrip() + " " + sql_text[where_match.end():]
                        return (sql_text[:where_match.start()] +
                                " WHERE\n  " + "\n  AND ".join(good) +
                                sql_text[where_match.end():])
                    sql_code = _strip_bad_where_conditions(sql_code)

                    # -- Fix bad alias.col refs: correct alias if possible, else NULL --
                    # Only apply on flat JOINs (no subqueries). When a model generates
                    # FROM (...) A subqueries, alias 'A' refers to the subquery result  - 
                    # not directly to the base table - so column-level validation is
                    # unreliable and replacing valid refs with NULL corrupts the SQL.
                    _has_subquery = bool(re.search(r'FROM\s*\(', sql_code, re.IGNORECASE))
                    if _alias_to_cols and not _has_subquery:
                        def _fix_col_ref(m: re.Match) -> str:
                            alias, col = m.group(1), m.group(2)
                            if _col_ref_valid(alias, col):
                                return m.group(0)  # already valid
                            col_lower = col.lower()
                            # Try to correct to the right table alias
                            for other_alias, other_cols in _alias_to_cols.items():
                                if col_lower in other_cols:
                                    correct = f"{other_alias.lower()}.{col}"
                                    _debug_status(f"[SQL-DEBUG] Corrected wrong-alias '{alias}.{col}'  -> '{correct}'")
                                    return correct
                            # Not in any table - replace with NULL
                            _debug_status(f"[SQL-DEBUG] Replaced non-existent column '{alias}.{col}' with NULL")
                            return 'NULL'
                        sql_code = re.sub(r'\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b',
                                          _fix_col_ref, sql_code)

                    _debug_status(f"[SQL-DEBUG] After clean_sql (len={len(sql_code)}): {sql_code[:300]!r}")

                    validation_error, sql_code = validate_sql(sql_code)

                    if not validation_error:
                        _debug_status(f"[SQL-DEBUG] Attempt {attempt+1}: PASSED validation")
                        break

                    msg = f"[SQL] Attempt {attempt+1} failed: {validation_error}"
                    status_messages.append(msg)
                    print(msg)
                    errors_so_far.append({"attempt": attempt, "sql": sql_code, "error": validation_error})

                    # Detect if model is stuck: same error 3 times in a row  -> give up
                    if len(errors_so_far) >= 3:
                        last_3 = [e["error"] for e in errors_so_far[-3:]]
                        if last_3[0] == last_3[1] == last_3[2]:
                            _debug_status(f"[SQL-DEBUG] Model stuck: same error 3 times in a row, stopping.")
                            model_stuck = True
                            break

                else:
                    # Safety ceiling reached
                    last_err = errors_so_far[-1]["error"] if errors_so_far else "Unknown"
                    _debug_status(f"[SQL-DEBUG] Safety ceiling ({MAX_SQL_ATTEMPTS}) reached. Last error: {last_err}")
                    retry_exhausted = True

                # Check if we broke out of the loop due to retries/stuck (not success)
                if validation_error:
                    last_err = errors_so_far[-1]["error"] if errors_so_far else "Unknown"
                    _debug_status(
                        f"[SQL-DEBUG] Retry limit reached (stuck={model_stuck}, exhausted={retry_exhausted}). "
                        "Attempting fallback with last generated SQL and structural validation only."
                    )
                    fallback_sql = errors_so_far[-1]["sql"] if errors_so_far else sql_code
                    fb_err, fb_sql = validate_sql(fallback_sql, semantic=False)

                    if fb_err and "Table with name" in fb_err and "does not exist" in fb_err:
                        repaired_sql, missing_replacements = _repair_missing_table_errors(fallback_sql, fb_err)
                        if missing_replacements:
                            _debug_status(f"[SQL] Fallback missing-table recovery mapping applied: {missing_replacements}")
                            fb_err, fb_sql = validate_sql(repaired_sql, semantic=False)

                    if fb_err:
                        return {
                            "error": (
                                f"SQL generation failed after {len(errors_so_far)} attempts; "
                                f"fallback validation also failed: {fb_err}"
                            )
                        }

                    sql_code = fb_sql
                    validation_error = None
                    msg = "[SQL] Fallback enabled: executing last generated query after retry limit"
                    status_messages.append(msg)
                    print(msg)

            msg = f"[SQL] Generated query: {sql_code[:200]}..."
            status_messages.append(msg)
            print(msg)

            if bridge.stop_generation_flag:
                return _stopped_result()
            
            # Print full SQL query to terminal before execution
            print("\n" + "="*80)
            print("[SQL] FULL QUERY TO BE EXECUTED:")
            print("="*80)
            print(sql_code)
            print("="*80 + "\n")
            
            # Also add to UI
            status_messages.append("\n" + "="*80)
            status_messages.append("[SQL] FULL QUERY TO BE EXECUTED:")
            status_messages.append("="*80)
            status_messages.append(sql_code)
            status_messages.append("="*80 + "\n")
            
            # Execute SQL query
            try:
                if bridge.stop_generation_flag:
                    conn.close()
                    return _stopped_result()
                msg = "[SQL] Executing query..."
                status_messages.append(msg)
                print(msg)
                
                result_df = conn.execute(sql_code).fetchdf()
                row_count = len(result_df)
                msg = f"[SQL] OK Query executed successfully: {row_count} rows returned"
                status_messages.append(msg)
                print(msg)
                
                # Save SQL result in requested format
                import pandas as pd
                msg = "[SQL] Converting to DataFrame..."
                status_messages.append(msg)
                print(msg)
                
                output_dir = os.path.join(app_data_path(), "processed_files")
                os.makedirs(output_dir, exist_ok=True)
                timestamp = int(time.time())

                def _extract_recon_splits(frame: "pd.DataFrame") -> dict[str, "pd.DataFrame"]:
                    """Return deterministic reconciliation buckets when recognizable."""
                    splits = {"All_Results": frame}
                    cols = {str(c).strip().lower(): c for c in frame.columns}

                    # Preferred split key: _merge from outer joins
                    if "_merge" in cols:
                        merge_col = cols["_merge"]
                        merge_norm = frame[merge_col].astype(str).str.strip().str.lower()
                        splits["Matched"] = frame[merge_norm == "both"]
                        splits["Missing_From_Second"] = frame[merge_norm.str.contains("left", na=False)]
                        splits["Missing_From_First"] = frame[merge_norm.str.contains("right", na=False)]
                        return splits

                    # Fallback split key: status-like category columns
                    for candidate in ("recon_status", "comparison_status", "status", "match_status"):
                        if candidate in cols:
                            status_col = cols[candidate]
                            status_norm = frame[status_col].astype(str).str.strip().str.lower()
                            splits["Matched"] = frame[status_norm.str.contains("match", na=False)]
                            splits["Mismatched"] = frame[status_norm.str.contains("mismatch|diff|different", regex=True, na=False)]
                            return splits

                    return splits

                wants_multi_excel = (
                    output_format == "excel"
                    and is_reconciliation
                    and any(k in instr_lower for k in ["multiple sheet", "multi sheet", "multi-sheet", "single excel", "one excel"])
                )
                wants_multi_csv = (
                    output_format == "csv"
                    and is_reconciliation
                    and any(k in instr_lower for k in ["multiple csv", "multi csv", "separate csv", "multiple output", "multi output", "multiple outputs"])
                )

                # Determine file extension based on output_format
                ext = "" if output_format == "none" else output_extension(output_format, default="xlsx")
                output_file = os.path.join(output_dir, f"analysis_{timestamp}.{ext}") if ext else ""

                if ext:
                    msg = f"[SQL] Saving to {ext.upper()} file..."
                    status_messages.append(msg)
                    print(msg)
                else:
                    msg = "[SQL] UI-only mode: no file export requested."
                    status_messages.append(msg)
                    print(msg)

                saved_paths = []
                if wants_multi_excel:
                    split_frames = _extract_recon_splits(result_df)
                    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
                        for sname, sdf in split_frames.items():
                            if sdf is None:
                                continue
                            safe_sheet = re.sub(r"[^A-Za-z0-9_ ]", "_", sname)[:31] or "Sheet1"
                            sdf.to_excel(writer, sheet_name=safe_sheet, index=False)
                    saved_paths.append(output_file)
                    msg = f"[SQL] OK Saved multi-sheet Excel ({len(split_frames)} sheets): {output_file}"
                    status_messages.append(msg)
                    print(msg)
                elif wants_multi_csv:
                    split_frames = _extract_recon_splits(result_df)
                    csv_dir = os.path.join(output_dir, f"analysis_{timestamp}_csv_parts")
                    os.makedirs(csv_dir, exist_ok=True)
                    for sname, sdf in split_frames.items():
                        if sdf is None:
                            continue
                        fname = re.sub(r"[^A-Za-z0-9_\-]", "_", sname).strip("_") or "part"
                        fpath = os.path.join(csv_dir, f"{fname}.csv")
                        sdf.to_csv(fpath, index=False)
                        saved_paths.append(fpath)
                    output_file = csv_dir
                    msg = f"[SQL] OK Saved multi-CSV output ({len(saved_paths)} files): {csv_dir}"
                    status_messages.append(msg)
                    print(msg)
                elif ext == "":
                    # UI-only mode: skip file generation.
                    saved_paths = []
                elif ext == "xlsx":
                    save_result = save_analysis_dataframe(result_df, output_dir, output_format, timestamp=timestamp)
                    output_file = save_result["file_path"]
                    saved_paths.append(output_file)
                elif ext == "csv":
                    save_result = save_analysis_dataframe(result_df, output_dir, output_format, timestamp=timestamp)
                    output_file = save_result["file_path"]
                    saved_paths.append(output_file)
                elif ext == "pdf":
                    save_result = save_analysis_dataframe(result_df, output_dir, output_format, timestamp=timestamp)
                    output_file = save_result["file_path"]
                    saved_paths.append(output_file)
                else:  # txt
                    save_result = save_analysis_dataframe(result_df, output_dir, output_format, timestamp=timestamp)
                    output_file = save_result["file_path"]
                    saved_paths.append(output_file)

                if ext and not wants_multi_excel and not wants_multi_csv:
                    msg = f"[SQL] OK Saved results to: {output_file}"
                    status_messages.append(msg)
                    print(msg)
                
                # Return status messages with count and file path
                summary = _clean_summary(status_messages, f"Total rows: {row_count}")
                
                conn.close()
                return {
                    "ok": True,
                    "response_text": summary,
                    "file_path": output_file if ext else None,
                    "file_paths": saved_paths,
                }
            except Exception as e:
                conn.close()
                return {"error": f"SQL execution failed: {str(e)}"}
            
        except Exception as e:
            print(f"[SQL] Error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"SQL analysis failed: {str(e)}"}

