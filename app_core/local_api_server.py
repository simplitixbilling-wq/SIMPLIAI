import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from app_core.local_secret_store import LocalSecretStore
from app_core.security_utils import (
    LOCAL_API_MAX_BODY_BYTES,
    is_allowed_local_origin,
    is_loopback_host,
    parse_content_length,
    safe_local_path,
)
from app_core.utils import app_data_path


class LocalApiServer:
    """Small localhost JSON API for external app integrations.

    Security model:
    - Binds to 127.0.0.1 only.
    - Requires X-API-Key for all /v1/* routes.
    """

    def __init__(self, bridge, host="127.0.0.1", port=8765):
        if not is_loopback_host(host):
            raise ValueError("Local API may only bind to a loopback host")
        self.bridge = bridge
        self.host = host
        self.port = int(port)
        self.server = None
        self.thread = None
        self.request_lock = threading.Lock()
        self.secret_store = LocalSecretStore(
            app_settings=getattr(self.bridge, "app_settings", {}),
            persist_settings=lambda settings: self.bridge.save_app_settings(settings),
            app_data_dir=app_data_path(""),
        )
        self.api_key, self.key_file = self.secret_store.ensure_local_api_key()

    def _check_api_key(self, headers):
        got = headers.get("X-API-Key", "")
        return self.secret_store.check_secret(got, self.api_key)

    @staticmethod
    def _to_text(value):
        if value is None:
            return ""
        return str(value)

    def _excel_rows_to_prompt(self, sheet_name, headers, rows, max_rows=None, max_cols=None, max_cell_chars=120):
        safe_sheet = self._to_text(sheet_name) or "Sheet1"
        hdr = [self._to_text(h).strip()[:max_cell_chars] for h in (headers or [])]
        data_rows = rows or []

        # Optional clipping (None means no explicit limit).
        clipped_rows = data_rows if max_rows is None else data_rows[:max_rows]
        if hdr and max_cols is not None:
            hdr = hdr[:max_cols]

        normalized = []
        for r in clipped_rows:
            if not isinstance(r, list):
                continue
            row_src = r if max_cols is None else r[:max_cols]
            row = [self._to_text(c).strip()[:max_cell_chars] for c in row_src]
            if row:
                normalized.append(row)

        if not hdr and normalized:
            width = max(len(r) for r in normalized)
            hdr = [f"Col{i+1}" for i in range(width)]

        # Pad rows to header width for stable tabular rendering.
        width = len(hdr)
        if width <= 0:
            return f"Sheet: {safe_sheet}\nNo usable table data was provided."
        for i, r in enumerate(normalized):
            if len(r) < width:
                normalized[i] = r + [""] * (width - len(r))

        lines = [f"Sheet: {safe_sheet}", f"Rows sent: {len(normalized)}", "", "Table:"]
        lines.append(" | ".join(hdr))
        lines.append(" | ".join(["---"] * width))
        for r in normalized:
            lines.append(" | ".join(r))
        table_text = "\n".join(lines)
        # Keep prompt size bounded for model stability.
        if len(table_text) > 24000:
            table_text = table_text[:24000] + "\n...[table truncated]"
        return table_text

    def _excel_sheets_to_prompt(self, sheets, max_cell_chars=120):
        """Build a combined workbook prompt from multiple sheets."""
        if not isinstance(sheets, list) or not sheets:
            return "No usable workbook data was provided.", 0, 0, 0

        blocks = []
        total_rows = 0
        max_cols = 0
        used_sheets = 0

        for s in sheets:
            if not isinstance(s, dict):
                continue
            sheet_name = s.get("sheet_name", "Sheet")
            headers = s.get("headers", [])
            rows = s.get("rows", [])
            if not isinstance(rows, list) or not rows:
                continue

            block = self._excel_rows_to_prompt(
                sheet_name=sheet_name,
                headers=headers,
                rows=rows,
                max_rows=None,
                max_cols=None,
                max_cell_chars=max_cell_chars,
            )
            blocks.append(block)
            total_rows += len(rows)
            if isinstance(headers, list):
                max_cols = max(max_cols, len(headers))
            used_sheets += 1

        if not blocks:
            return "No usable workbook data was provided.", 0, 0, 0

        text = "\n\n".join(blocks)
        if len(text) > 80000:
            text = text[:80000] + "\n...[workbook truncated]"
        return text, used_sheets, total_rows, max_cols

    def _excel_action_prompt(self, action: str, instruction: str):
        """Return (role, task, steps) tuned for each Excel action type."""

        base_role = "You are an expert data analyst and Excel specialist."

        if action == "chart":
            return (
                base_role,
                "Recommend the best chart for this data. Return ONLY a JSON block with keys: "
                "chart_type (one of: bar, line, pie, column, scatter, area), "
                "title (string), x_column (header name for X axis), "
                "y_columns (array of header names for Y values), "
                "insight (one sentence why this chart). "
                "Do NOT include any text outside the JSON block.",
                "1. Examine headers and data types.\n"
                "2. Pick the most insightful chart type.\n"
                "3. Return ONLY valid JSON, no markdown fences."
            )

        if action == "predict":
            return (
                base_role,
                "Forecast the next 5 values based on the trend in the data. Return ONLY a JSON block with keys: "
                "column (header name being predicted), "
                "values (array of 5 predicted numbers), "
                "method (e.g. 'linear trend', 'moving average'), "
                "confidence (low/medium/high), "
                "explanation (one sentence). "
                "Do NOT include any text outside the JSON block.",
                "1. Identify numeric columns with trends.\n"
                "2. Calculate projected values.\n"
                "3. Return ONLY valid JSON, no markdown fences."
            )

        if action == "anomalies":
            return (
                base_role,
                "Find anomalies/outliers in the data. Return ONLY a JSON block with keys: "
                "anomalies (array of objects, each with: row (1-based data row number), "
                "column (header name), value (the anomalous value), "
                "reason (why it is anomalous)), "
                "summary (one sentence overview). "
                "Do NOT include any text outside the JSON block.",
                "1. Examine each numeric column for statistical outliers.\n"
                "2. Check for missing, duplicate, or inconsistent values.\n"
                "3. Return ONLY valid JSON, no markdown fences."
            )

        if action == "format":
            return (
                base_role,
                "Suggest conditional formatting rules for this data. Return ONLY a JSON block with keys: "
                "rules (array of objects, each with: column (header name), "
                "condition (e.g. 'greater_than', 'less_than', 'equals', 'contains', 'top_n', 'bottom_n'), "
                "threshold (value), color (red/yellow/green/blue/orange)), "
                "summary (one sentence). "
                "Do NOT include any text outside the JSON block.",
                "1. Identify columns that benefit from visual highlighting.\n"
                "2. Pick meaningful thresholds from the data.\n"
                "3. Return ONLY valid JSON, no markdown fences."
            )

        if action == "formula":
            return (
                base_role,
                "Generate Excel formulas for the user's request. Return ONLY a JSON block with keys: "
                "formulas (array of objects, each with: cell (e.g. 'I2'), "
                "formula (Excel formula string starting with =), "
                "description (what it calculates)), "
                "fill_down (true/false — should formulas be filled down for all rows). "
                "Do NOT include any text outside the JSON block.",
                "1. Map user request to Excel formula syntax.\n"
                "2. Use actual column letters based on the headers provided.\n"
                "3. Return ONLY valid JSON, no markdown fences."
            )

        if action == "clean":
            return (
                base_role,
                "Identify data quality issues and suggest fixes. Return ONLY a JSON block with keys: "
                "issues (array of objects, each with: row (1-based), column (header name), "
                "current_value, suggested_value, issue_type (e.g. 'missing', 'typo', 'format', 'duplicate')), "
                "summary (one sentence overview). "
                "Do NOT include any text outside the JSON block.",
                "1. Scan for missing, inconsistent, duplicate values.\n"
                "2. Suggest corrected values where possible.\n"
                "3. Return ONLY valid JSON, no markdown fences."
            )

        if action == "summary":
            return (
                base_role,
                "Create a comprehensive summary dashboard of this data with statistics. "
                "Return ONLY a JSON block with keys: "
                "total_rows (number), columns_analyzed (array of header names), "
                "stats (array of objects, each with: column, min, max, avg, sum, unique_count), "
                "top_insights (array of 3-5 short insight strings), "
                "data_quality_score (0-100). "
                "Do NOT include any text outside the JSON block.",
                "1. Compute statistics for all numeric columns.\n"
                "2. Identify top insights.\n"
                "3. Return ONLY valid JSON, no markdown fences."
            )

        # Default: analyze (text output)
        return (
            base_role,
            "Analyze the provided worksheet range and answer the user instruction with clear, actionable output. "
            "Compute totals/trends directly from the provided table and mention any data quality concerns.",
            "1. Parse rows and columns carefully.\n"
            "2. Compute requested metrics from the table.\n"
            "3. Return concise results with bullet points.\n"
            "4. If data is insufficient, state exactly what is missing."
        )

    @staticmethod
    def _json_response(handler, status_code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        origin = handler.headers.get("Origin", "")
        handler.send_response(status_code)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        if is_allowed_local_origin(origin):
            handler.send_header("Access-Control-Allow-Origin", origin or "http://127.0.0.1")
            handler.send_header("Vary", "Origin")
            handler.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
            handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            handler.send_header("Access-Control-Max-Age", "600")
        handler.end_headers()
        handler.wfile.write(body)

    @staticmethod
    def _allowed_file_roots():
        roots = [app_data_path("")]
        home = os.path.expanduser("~")
        if home:
            roots.append(home)
        return roots

    def _safe_jsonl_path(self, path: str, *, must_exist: bool = False) -> str:
        return safe_local_path(path, allowed_roots=self._allowed_file_roots(), must_exist=must_exist)

    def _sanitize_file_payloads(self, files):
        sanitized = []
        for item in files:
            if isinstance(item, str):
                safe_path = self._safe_jsonl_path(item, must_exist=True)
                sanitized.append({
                    "name": os.path.basename(safe_path),
                    "path": safe_path,
                    "size": os.path.getsize(safe_path),
                })
                continue
            if not isinstance(item, dict):
                raise ValueError("Each file must be an object or path string")
            copied = dict(item)
            path = str(copied.get("path", "")).strip()
            if path:
                safe_path = self._safe_jsonl_path(path, must_exist=True)
                copied["path"] = safe_path
                copied.setdefault("name", os.path.basename(safe_path))
                copied.setdefault("size", os.path.getsize(safe_path))
            sanitized.append(copied)
        return sanitized

    def _build_handler(self):
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Keep server quiet unless needed.
                return

            def do_OPTIONS(self):
                if not is_allowed_local_origin(self.headers.get("Origin", "")):
                    api._json_response(self, 403, {"error": "Origin not allowed"})
                    return
                api._json_response(self, 200, {"ok": True})

            def do_GET(self):
                path = urlparse(self.path).path

                if path == "/health":
                    model_status = {}
                    try:
                        model_status = api.bridge.get_model_status()
                    except Exception:
                        model_status = {}
                    api._json_response(self, 200, {
                        "ok": True,
                        "service": "SIMPLE_AI Local API",
                        "host": api.host,
                        "port": api.port,
                        "model_loaded": bool(model_status.get("loaded")),
                    })
                    return

                if path == "/v1/info":
                    if not api._check_api_key(self.headers):
                        api._json_response(self, 401, {"error": "Unauthorized"})
                        return
                    try:
                        info = api.bridge.get_app_info()
                        api._json_response(self, 200, {
                            "ok": True,
                            "app": "SIMPLE_AI",
                            "model": {
                                "loaded": info.get("model_loaded", False),
                                "name": info.get("model_name", ""),
                                "n_ctx": info.get("n_ctx", 0),
                            },
                            "mode": (info.get("config") or {}).get("mode", ""),
                        })
                    except Exception as e:
                        api._json_response(self, 500, {"error": str(e)})
                    return

                api._json_response(self, 404, {"error": "Not found"})

            def do_POST(self):
                path = urlparse(self.path).path

                if path.startswith("/v1/") and not api._check_api_key(self.headers):
                    api._json_response(self, 401, {"error": "Unauthorized"})
                    return

                length, length_error = parse_content_length(
                    self.headers.get("Content-Length", 0),
                    max_bytes=LOCAL_API_MAX_BODY_BYTES,
                )
                if length_error:
                    status = 413 if "too large" in length_error.lower() else 400
                    api._json_response(self, status, {"error": length_error})
                    return
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8")) if raw else {}
                except Exception:
                    api._json_response(self, 400, {"error": "Invalid JSON body"})
                    return

                if path == "/v1/chat":
                    text = str(payload.get("text", "")).strip()
                    role = str(payload.get("role", ""))
                    task = str(payload.get("task", ""))
                    steps = str(payload.get("steps", ""))
                    output_format = str(payload.get("output_format", "none")).strip().lower()
                    if not text:
                        api._json_response(self, 400, {"error": "'text' is required"})
                        return

                    with api.request_lock:
                        result = api.bridge.agent_chat(text=text, role=role, task=task, steps=steps, output_format=output_format)

                    status = 200 if not result.get("error") else 400
                    api._json_response(self, status, result)
                    return

                if path == "/v1/process-files":
                    files = payload.get("files", [])
                    instructions = str(payload.get("instructions", "")).strip()
                    output_format = str(payload.get("output_format", "excel")).strip().lower()
                    if not isinstance(files, list) or not files:
                        api._json_response(self, 400, {"error": "'files' must be a non-empty array"})
                        return
                    if not instructions:
                        api._json_response(self, 400, {"error": "'instructions' is required"})
                        return
                    try:
                        files = api._sanitize_file_payloads(files)
                    except ValueError as e:
                        api._json_response(self, 400, {"error": f"Invalid files: {e}"})
                        return

                    with api.request_lock:
                        result = api.bridge.process_files_with_ai(
                            files=files,
                            instructions=instructions,
                            output_format=output_format,
                        )

                    status = 200 if not result.get("error") else 400
                    api._json_response(self, status, result)
                    return

                if path == "/v1/excel-range":
                    instruction = str(payload.get("instruction", "")).strip()
                    sheet_name = payload.get("sheet_name", "Sheet1")
                    headers = payload.get("headers", [])
                    rows = payload.get("rows", [])
                    sheets = payload.get("sheets", [])
                    action = str(payload.get("action", "analyze")).strip().lower()

                    if not instruction:
                        api._json_response(self, 400, {"error": "'instruction' is required"})
                        return

                    use_multi_sheet = isinstance(sheets, list) and len(sheets) > 0
                    if use_multi_sheet:
                        table_text, sheets_used, rows_used, cols_used = api._excel_sheets_to_prompt(sheets)
                        if rows_used <= 0:
                            api._json_response(self, 400, {"error": "'sheets' must include at least one sheet with non-empty rows"})
                            return
                    else:
                        if not isinstance(rows, list) or not rows:
                            api._json_response(self, 400, {"error": "'rows' must be a non-empty array"})
                            return
                        table_text = api._excel_rows_to_prompt(sheet_name, headers, rows, max_rows=None, max_cols=None)
                        sheets_used = 1
                        rows_used = len(rows)
                        cols_used = len(headers) if isinstance(headers, list) else 0

                    role, task, steps = api._excel_action_prompt(action, instruction)
                    chat_text = f"Instruction: {instruction}\n\n{table_text}"

                    with api.request_lock:
                        result = api.bridge.agent_chat(
                            text=chat_text,
                            role=role,
                            task=task,
                            steps=steps,
                        )

                    # Retry in compact mode if the model returned empty output.
                    if result.get("error") and "No response from model" in str(result.get("error", "")):
                        if use_multi_sheet:
                            compact_blocks = []
                            for s in sheets[:3]:
                                if not isinstance(s, dict):
                                    continue
                                compact_blocks.append(api._excel_rows_to_prompt(
                                    s.get("sheet_name", "Sheet"),
                                    s.get("headers", []),
                                    s.get("rows", []),
                                    max_rows=40,
                                    max_cols=12,
                                    max_cell_chars=80,
                                ))
                            compact = "\n\n".join(compact_blocks)
                        else:
                            compact = api._excel_rows_to_prompt(
                                sheet_name,
                                headers,
                                rows,
                                max_rows=60,
                                max_cols=12,
                                max_cell_chars=80,
                            )
                        compact_text = (
                            f"Instruction: {instruction}\n\n"
                            f"Use this compact table sample if the full table is too large:\n{compact}"
                        )
                        with api.request_lock:
                            result = api.bridge.agent_chat(
                                text=compact_text,
                                role=role,
                                task=task,
                                steps=steps,
                            )

                    if result.get("error"):
                        api._json_response(self, 400, result)
                        return

                    api._json_response(self, 200, {
                        "ok": True,
                        "text": result.get("text", ""),
                        "action": action,
                        "sheet_name": api._to_text(sheet_name) or "Sheet1",
                        "sheets_used": sheets_used,
                        "rows_received": rows_used,
                        "rows_used": rows_used,
                        "columns_used": cols_used,
                    })
                    return

                if path == "/v1/batch-jsonl":
                    jsonl_content = str(payload.get("jsonl", "")).strip()
                    jsonl_path = str(payload.get("jsonl_path", "")).strip()
                    checkpoint_dir = str(payload.get("checkpoint_dir", "")).strip()
                    try:
                        checkpoint_every = int(payload.get("checkpoint_every", 12))
                    except (TypeError, ValueError):
                        checkpoint_every = 12

                    if not jsonl_content and not jsonl_path:
                        api._json_response(self, 400, {"error": "'jsonl' or 'jsonl_path' is required"})
                        return

                    if jsonl_path:
                        try:
                            jsonl_path = api._safe_jsonl_path(jsonl_path, must_exist=True)
                        except ValueError as e:
                            api._json_response(self, 400, {"error": f"Invalid jsonl_path: {e}"})
                            return
                    if checkpoint_dir:
                        try:
                            checkpoint_dir = api._safe_jsonl_path(checkpoint_dir, must_exist=False)
                        except ValueError as e:
                            api._json_response(self, 400, {"error": f"Invalid checkpoint_dir: {e}"})
                            return

                    with api.request_lock:
                        result = api.bridge.run_jsonl_queue(
                            jsonl=jsonl_content,
                            jsonl_path=jsonl_path,
                            checkpoint_dir=checkpoint_dir,
                            checkpoint_every=checkpoint_every,
                        )

                    status = 200 if result.get("ok") else 400
                    api._json_response(self, status, result)
                    return

                if path == "/v1/stop":
                    with api.request_lock:
                        result = api.bridge.stop_generation()
                    api._json_response(self, 200, result)
                    return

                api._json_response(self, 404, {"error": "Not found"})

        return Handler

    def start(self):
        if self.server is not None:
            return {
                "ok": True,
                "host": self.host,
                "port": self.port,
                "api_key_file": self.key_file,
            }

        handler = self._build_handler()
        self.server = ThreadingHTTPServer((self.host, self.port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        return {
            "ok": True,
            "host": self.host,
            "port": self.port,
            "api_key_file": self.key_file,
        }

    def stop(self):
        if self.server is None:
            return
        try:
            self.server.shutdown()
            self.server.server_close()
        finally:
            self.server = None
            self.thread = None
