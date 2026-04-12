import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from utils import app_data_path


class LocalApiServer:
    """Small localhost JSON API for external app integrations.

    Security model:
    - Binds to 127.0.0.1 only.
    - Requires X-API-Key for all /v1/* routes.
    """

    def __init__(self, bridge, host="127.0.0.1", port=8765):
        self.bridge = bridge
        self.host = host
        self.port = int(port)
        self.server = None
        self.thread = None
        self.request_lock = threading.Lock()
        self.key_file = app_data_path("local_api_key.txt")
        self.api_key = self._ensure_api_key()

    def _ensure_api_key(self):
        key = ""
        try:
            key = str((self.bridge.app_settings or {}).get("local_api_key", "")).strip()
        except Exception:
            key = ""

        if not key:
            key = secrets.token_urlsafe(24)
            try:
                self.bridge.save_app_settings({"local_api_key": key})
            except Exception:
                pass

        # Write key to local file for easy retrieval from external tools.
        try:
            with open(self.key_file, "w", encoding="utf-8") as f:
                f.write(key)
        except Exception:
            pass

        return key

    def _check_api_key(self, headers):
        got = headers.get("X-API-Key", "")
        return bool(got and got == self.api_key)

    @staticmethod
    def _to_text(value):
        if value is None:
            return ""
        return str(value)

    def _excel_rows_to_prompt(self, sheet_name, headers, rows, max_rows=200, max_cols=20, max_cell_chars=120):
        safe_sheet = self._to_text(sheet_name) or "Sheet1"
        hdr = [self._to_text(h).strip()[:max_cell_chars] for h in (headers or [])]
        data_rows = rows or []

        # Keep request bounded for predictable latency and token usage.
        clipped_rows = data_rows[:max_rows]
        if hdr:
            hdr = hdr[:max_cols]

        normalized = []
        for r in clipped_rows:
            if not isinstance(r, list):
                continue
            row = [self._to_text(c).strip()[:max_cell_chars] for c in r[:max_cols]]
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

    @staticmethod
    def _json_response(handler, status_code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        handler.send_response(status_code)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
        handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        handler.end_headers()
        handler.wfile.write(body)

    def _build_handler(self):
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Keep server quiet unless needed.
                return

            def do_OPTIONS(self):
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

                length = int(self.headers.get("Content-Length", 0) or 0)
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
                    if not text:
                        api._json_response(self, 400, {"error": "'text' is required"})
                        return

                    with api.request_lock:
                        result = api.bridge.agent_chat(text=text, role=role, task=task, steps=steps)

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

                    if not instruction:
                        api._json_response(self, 400, {"error": "'instruction' is required"})
                        return
                    if not isinstance(rows, list) or not rows:
                        api._json_response(self, 400, {"error": "'rows' must be a non-empty array"})
                        return

                    table_text = api._excel_rows_to_prompt(sheet_name, headers, rows)
                    role = str(payload.get("role", "You are an expert spreadsheet analyst.")).strip()
                    task = (
                        "Analyze the provided worksheet range and answer the user instruction with clear, actionable output. "
                        "Compute totals/trends directly from the provided table and mention any data quality concerns."
                    )
                    steps = (
                        "1. Parse rows and columns carefully.\n"
                        "2. Compute requested metrics from the table.\n"
                        "3. Return concise results with bullet points.\n"
                        "4. If data is insufficient, state exactly what is missing."
                    )
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
                        "sheet_name": api._to_text(sheet_name) or "Sheet1",
                        "rows_received": len(rows),
                        "rows_used": min(len(rows), 200),
                        "columns_used": min(len(headers) if isinstance(headers, list) else 0, 20),
                    })
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
                "api_key": self.api_key,
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
            "api_key": self.api_key,
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
