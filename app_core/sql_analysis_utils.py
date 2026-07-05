"""Pure SQL text helpers for analysis workflows."""

from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Callable


def normalize_table_token(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def extract_direct_sql(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    fence = re.search(r"```(?:sql)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        if re.match(r"^(select|with)\b", candidate, re.IGNORECASE) and re.search(r"\bfrom\b", candidate, re.IGNORECASE):
            return candidate

    match = re.search(r"\b(select|with)\b[\s\S]*", raw, flags=re.IGNORECASE)
    if match:
        candidate = match.group(0).strip()
        if re.match(r"^(select|with)\b", candidate, re.IGNORECASE) and re.search(r"\bfrom\b", candidate, re.IGNORECASE):
            return candidate

    return ""


def clean_sql_text(raw_sql: str, debug: Callable[[str], None] | None = None) -> str:
    emit = debug or (lambda _msg: None)
    emit(f"[SQL-DEBUG] clean_sql input (len={len(raw_sql)}):\n{raw_sql!r}")
    if not raw_sql:
        emit("[SQL-DEBUG] clean_sql: input is empty, returning ''")
        return ""

    text = raw_sql.strip()
    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
        emit(f"[SQL-DEBUG] clean_sql: extracted from fence block: {text[:200]!r}")

    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    text = "\n".join(lines).strip()
    emit(f"[SQL-DEBUG] clean_sql output (len={len(text)}): {text[:200]!r}")
    return text


def strip_non_sql_prefix(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    raw = re.sub(r"^\s*query\s*[-_ ]*\d+\s*:\s*", "", raw, flags=re.IGNORECASE)
    match = re.search(r"\b(select|with)\b", raw, flags=re.IGNORECASE)
    if match and match.start() > 0:
        raw = raw[match.start():]
    return raw.strip()


def split_sql_statements(text: str) -> list[str]:
    """Split SQL by semicolons outside strings/comments."""
    parts = []
    buff = []
    i = 0
    n = len(text or "")
    state = "normal"
    src = text or ""

    while i < n:
        ch = src[i]
        nxt = src[i + 1] if i + 1 < n else ""

        if state == "line_comment":
            if ch == "\n":
                state = "normal"
                buff.append(ch)
            i += 1
            continue

        if state == "block_comment":
            if ch == "*" and nxt == "/":
                state = "normal"
                i += 2
                continue
            i += 1
            continue

        if state == "s_quote":
            buff.append(ch)
            if ch == "'":
                if nxt == "'":
                    buff.append(nxt)
                    i += 2
                    continue
                state = "normal"
            i += 1
            continue

        if state == "d_quote":
            buff.append(ch)
            if ch == '"':
                if nxt == '"':
                    buff.append(nxt)
                    i += 2
                    continue
                state = "normal"
            i += 1
            continue

        if ch == "-" and nxt == "-":
            state = "line_comment"
            i += 2
            continue
        if ch == "/" and nxt == "*":
            state = "block_comment"
            i += 2
            continue
        if ch == "'":
            state = "s_quote"
            buff.append(ch)
            i += 1
            continue
        if ch == '"':
            state = "d_quote"
            buff.append(ch)
            i += 1
            continue
        if ch == ";":
            stmt = "".join(buff).strip()
            if stmt:
                parts.append(stmt)
            buff = []
            i += 1
            continue

        buff.append(ch)
        i += 1

    tail = "".join(buff).strip()
    if tail:
        parts.append(tail)
    return parts


def validate_sql_basic_structure(sql_text: str, table_names) -> str | None:
    """Return an error for basic read-only SELECT/WITH structure violations."""
    normalized = sql_text or ""
    if re.search(r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|pragma|copy)\b", normalized, re.IGNORECASE):
        return "Non-read-only SQL detected"
    if not re.match(r"^(select|with)\b", normalized, re.IGNORECASE):
        return "SQL must start with SELECT or WITH"
    if not any(tbl in normalized for tbl in table_names):
        return "SQL does not reference available tables"
    return None


def validate_sql_table_usage(sql_text: str, table_names, is_reconciliation: bool = False) -> str | None:
    """Validate table reference count and JOIN ON predicate shape."""
    tables = list(table_names)
    normalized = sql_text or ""
    referenced_tables = [
        tbl for tbl in tables
        if re.search(rf"\b{re.escape(tbl)}\b", normalized, re.IGNORECASE)
    ]
    if is_reconciliation and len(tables) >= 2 and len(referenced_tables) < 2:
        return "Reconciliation requires using both input tables"

    join_match = re.search(
        r"\bfrom\s+([A-Za-z_][A-Za-z0-9_]*)\b[\s\S]*?\bjoin\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        normalized,
        re.IGNORECASE,
    )
    if is_reconciliation and len(tables) >= 2 and join_match:
        if join_match.group(1).lower() == join_match.group(2).lower():
            return "Join uses the same table on both sides; use both input files"

    on_clauses = re.findall(
        r"\bON\s+([\s\S]*?)(?=\b(?:JOIN|WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|UNION|EXCEPT|INTERSECT)\b|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    comparator_re = re.compile(
        r"(=|<>|!=|<=|>=|<|>|\bIS\s+NOT\s+DISTINCT\s+FROM\b|\bIS\s+DISTINCT\s+FROM\b|\bLIKE\b|\bILIKE\b|\bIN\b|\bBETWEEN\b)",
        flags=re.IGNORECASE,
    )
    for cond in on_clauses:
        cond_text = cond.strip()
        if cond_text and not comparator_re.search(cond_text):
            return f"Invalid JOIN ON condition (must be boolean): {cond_text[:120]}"

    return None


def apply_fuzzy_table_mapping(sql_text: str, table_names) -> tuple[str, dict]:
    """Map FROM/JOIN table tokens to known table names using aliases and fuzzy matching."""
    if not sql_text:
        return sql_text, {}

    available = list(table_names)
    available_lower = {t.lower(): t for t in available}
    available_norm = {normalize_table_token(t): t for t in available}
    ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]

    def build_alias_map() -> dict[str, str]:
        alias_map = {}
        generic_noise = {"df", "data", "dataset", "table", "records", "record", "sheet", "file"}
        for i, tbl in enumerate(available):
            candidates = set()
            raw = str(tbl)
            friendly = raw[3:] if raw.lower().startswith("df_") else raw
            friendly_parts = [p for p in re.split(r"[^a-z0-9]+", friendly.lower()) if p]
            candidates.update({raw, friendly})
            if i < len(ordinals):
                candidates.update({ordinals[i], f"file{i+1}", f"report{chr(ord('a') + i)}"})
            if friendly_parts:
                candidates.update({friendly_parts[0], friendly_parts[-1]})
                for part in friendly_parts:
                    if (len(part) >= 3 or re.fullmatch(r"\d+[a-z]?", part)) and part not in generic_noise:
                        candidates.add(part)
                filtered = [p for p in friendly_parts if p not in generic_noise]
                if filtered:
                    candidates.update({"_".join(filtered), "".join(filtered)})
            norm_tbl = normalize_table_token(friendly)
            if "books" in norm_tbl:
                candidates.update({"books", "book"})
            if "2a" in norm_tbl or "gstr2a" in norm_tbl:
                candidates.update({"2a", "gstr2a"})
            for cand in candidates:
                norm = normalize_table_token(cand)
                if norm and norm not in alias_map:
                    alias_map[norm] = tbl
        return alias_map

    alias_norm_map = build_alias_map()
    cte_names = {
        m.group(1).strip().lower()
        for m in re.finditer(r"\b(?:with|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s+as\s*\(", sql_text, flags=re.IGNORECASE)
    }
    books_table = next((t for t in available if "books" in normalize_table_token(t)), None)
    twoa_table = next((t for t in available if "2a" in normalize_table_token(t) or "gstr2a" in normalize_table_token(t)), None)
    replacements = {}

    refs = re.findall(r"\b(?:from|join)\s+([A-Za-z0-9_]+|\"[^\"]+\")", sql_text, flags=re.IGNORECASE)
    for ref in refs:
        token = ref.strip().strip('"')
        if token.lower() in cte_names:
            continue
        mapped = None
        token_norm = normalize_table_token(token)
        if token_norm in {"books", "book"} and books_table:
            mapped = books_table
        elif token_norm in {"2a", "gstr2a"} and twoa_table:
            mapped = twoa_table
        if mapped is None and token.lower() in available_lower:
            mapped = available_lower[token.lower()]
        elif mapped is None:
            if token_norm in available_norm:
                mapped = available_norm[token_norm]
            elif token_norm in alias_norm_map:
                mapped = alias_norm_map[token_norm]
            else:
                norm_space = list(set(list(available_norm.keys()) + list(alias_norm_map.keys())))
                best = get_close_matches(token_norm, norm_space, n=1, cutoff=0.55)
                if best:
                    mapped = available_norm.get(best[0]) or alias_norm_map.get(best[0])
        if mapped and mapped != token:
            replacements[token] = mapped

    remapped = sql_text
    for src, dst in replacements.items():
        remapped = re.sub(
            rf"(\b(?:from|join)\s+)(?:\"?{re.escape(src)}\"?)\b",
            rf"\1{dst}",
            remapped,
            flags=re.IGNORECASE,
        )
    return remapped, replacements


def repair_missing_table_errors(sql_text: str, error_text: str, table_names) -> tuple[str, dict]:
    if not sql_text or not error_text:
        return sql_text, {}
    missing_names = re.findall(r"Table with name\s+([A-Za-z_][A-Za-z0-9_]*)\s+does not exist", error_text, flags=re.IGNORECASE)
    if not missing_names:
        return sql_text, {}
    patched_sql = sql_text
    replacements = {}
    for missing in missing_names:
        probe_sql = re.sub(
            rf"(\b(?:from|join)\s+)(?:\"?{re.escape(missing)}\"?)\b",
            rf"\1{missing}",
            patched_sql,
            flags=re.IGNORECASE,
        )
        remapped, rep = apply_fuzzy_table_mapping(probe_sql, table_names)
        if rep:
            patched_sql = remapped
            replacements.update(rep)
    return patched_sql, replacements


def repair_sql_identifiers(sql_text: str, table_columns: dict, debug: Callable[[str], None] | None = None) -> str:
    """Repair common alias/identifier mistakes using known table columns."""
    emit = debug or (lambda _msg: None)
    if not sql_text:
        return sql_text

    repaired = sql_text
    alias_map = {}
    alias_pattern = re.compile(
        r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:as\s+)?([A-Za-z0-9_]+)\b",
        re.IGNORECASE,
    )
    for match in alias_pattern.finditer(repaired):
        source_table = match.group(1)
        alias = match.group(2)
        alias_map[alias] = (f"T{alias}" if re.match(r"^[0-9]", alias) else alias, source_table)

    for old_alias, (new_alias, _table) in alias_map.items():
        if old_alias != new_alias:
            repaired = re.sub(rf"\b{re.escape(old_alias)}\b", new_alias, repaired)
            emit(f"[SQL-DEBUG] repair_sql_identifiers: alias '{old_alias}' -> '{new_alias}'")

    for _old_alias, (alias, source_table) in alias_map.items():
        if source_table not in table_columns:
            continue
        cols = [str(c) for c in table_columns[source_table]]
        for col in cols:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", col):
                continue
            quoted = '"' + col.replace('"', '""') + '"'
            before = f"{alias}.{col}"
            after = f"{alias}.{quoted}"
            if before in repaired:
                repaired = repaired.replace(before, after)
                emit(f"[SQL-DEBUG] repair_sql_identifiers: quoted column ref {before!r} -> {after!r}")

        known_set = set(cols)
        pattern = re.compile(rf"\b{re.escape(alias)}\.([A-Za-z_][A-Za-z0-9_]*)\b")

        def fix_col(match):
            col = match.group(1)
            if col in known_set:
                return match.group(0)
            near = get_close_matches(col, cols, n=1, cutoff=0.82)
            if near:
                fixed = f"{alias}.{near[0]}"
                emit(f"[SQL-DEBUG] repair_sql_identifiers: corrected {alias}.{col} -> {fixed}")
                return fixed
            return match.group(0)

        repaired = pattern.sub(fix_col, repaired)

    return repaired


def repair_sql_join_predicates(sql_text: str, debug: Callable[[str], None] | None = None) -> str:
    """Repair common invalid JOIN ON patterns produced by LLMs."""
    emit = debug or (lambda _msg: None)
    if not sql_text:
        return sql_text
    repaired = re.sub(
        r"\bON\s+COALESCE\s*\(\s*([^,\)]+?)\s*,\s*([^\)]+?)\s*\)",
        r"ON \1 IS NOT DISTINCT FROM \2",
        sql_text,
        flags=re.IGNORECASE,
    )
    if repaired != sql_text:
        emit("[SQL-DEBUG] repair_sql_join_predicates: fixed ON COALESCE(...) to IS NOT DISTINCT FROM")
    return repaired
