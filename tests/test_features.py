"""
Feature tests for bridge.py:
  Test 1 – Table format prompt detection
  Test 2 – PDF export on summarization content
  Test 3 – Financial analysis content → Excel export
Run:  venv311_3.11/Scripts/python.exe tests/test_features.py
"""
import sys, os, json, textwrap, re, csv, time, traceback

# ── make bridge importable without launching the webview ──────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub `webview` so bridge.py doesn't crash on import
import types, unittest.mock as mock

webview_stub = types.ModuleType("webview")
webview_stub.create_window = mock.MagicMock()
webview_stub.start = mock.MagicMock()
webview_stub.windows = []
sys.modules.setdefault("webview", webview_stub)

# ── narrow import: only the helper methods we need ───────────────────────────
import importlib, inspect

# Pull only what we need from bridge without running main()
bridge_source = open(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app_core", "bridge.py"),
    encoding="utf-8"
).read()

# We instantiate Bridge via a minimal fake so no llama / webview is loaded
exec_ns: dict = {"__name__": "bridge_test"}
exec(compile(bridge_source, "bridge.py", "exec"), exec_ns)
Bridge = exec_ns["Bridge"]

# ── build a minimal Bridge that skips __init__ side-effects ──────────────────
b = object.__new__(Bridge)
b.window = None
b.chats = {}
b.model_path = None
b.llm = None
b.app_settings = {}
b.selected_rag = None
b.rag_manager = None
b.tts_engine = None
b.web_search_enabled = False

EXPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports", "test_run")
os.makedirs(EXPORTS_DIR, exist_ok=True)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f"  →  {detail}" if detail else ""))
    results.append((name, condition))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("TEST 1 – Table format prompt detection (_parse_prompt_options)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

table_prompts = [
    "Show the results in table format",
    "Give me a tabular comparison",
    "List them as a table",
    "Present in a markdown table",
]
non_table_prompts = [
    "Summarize the report",
    "What is the capital of France?",
]

for prompt in table_prompts:
    opts = b._parse_prompt_options(prompt)
    check(f'Table detected: "{prompt[:50]}"',
          opts["response_format"] == "table",
          f"got response_format={opts['response_format']!r}")

for prompt in non_table_prompts:
    opts = b._parse_prompt_options(prompt)
    check(f'No table for: "{prompt[:50]}"',
          opts["response_format"] == "normal",
          f"got response_format={opts['response_format']!r}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("TEST 2 – PDF export on summarization content")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

SUMMARY_TEXT = textwrap.dedent("""\
    ## Quarterly Summary – Q1 2026

    The company achieved a revenue of $4.2M in Q1 2026, a 12% increase year-over-year.
    Operating expenses were reduced by 8% due to automation initiatives.

    Key highlights:
    - Product sales: $2.8M (+15%)
    - Service revenue: $1.4M (+7%)
    - Net profit margin: 22% (up from 18%)

    Overall the quarter demonstrated strong growth and disciplined cost management.
""")

CHAT_ID = "test_summary_chat"
b.chats[CHAT_ID] = [
    {"role": "user",      "content": "Summarize Q1 2026 performance and export to PDF"},
    {"role": "assistant", "content": SUMMARY_TEXT},
]

try:
    # Verify prompt detection for PDF export
    opts = b._parse_prompt_options("Summarize Q1 2026 performance and export to PDF")
    check("Prompt: export_format detected as 'pdf'",
          opts["export_format"] == "pdf",
          f"got {opts['export_format']!r}")

    # Build payload for message index 1 (assistant reply)
    payload = b._get_export_payload(CHAT_ID, 1)
    check("Payload title not empty",    bool(payload["title"]))
    check("Payload text matches reply", SUMMARY_TEXT in payload["text"])

    # Write PDF
    pdf_path = os.path.join(EXPORTS_DIR, "test_summary_export.pdf")
    b._write_pdf_export(pdf_path, payload["title"], payload["text"])
    check("PDF file created",  os.path.isfile(pdf_path),  pdf_path)
    check("PDF size > 1 KB",   os.path.getsize(pdf_path) > 1024,
          f"{os.path.getsize(pdf_path)} bytes")

    # Verify it's a real PDF (starts with %PDF)
    with open(pdf_path, "rb") as f:
        header = f.read(4)
    check("PDF has valid header", header == b"%PDF", f"header={header!r}")

    print(f"  → PDF saved: {pdf_path}")

except Exception as exc:
    check("PDF export pipeline", False, traceback.format_exc(limit=2))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("TEST 3 – Financial analysis with 5-year planning → Excel export")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

FINANCIAL_ANALYSIS = textwrap.dedent("""\
    ## Financial Analysis & 10% Growth Plan – 2026-2030

    Based on 2025 revenue of $5,000,000, applying a 10% annual growth target:

    | Year | Revenue ($)  | Growth (%) | Operating Cost ($) | Net Profit ($) |
    |------|-------------|------------|-------------------|----------------|
    | 2026 | 5,500,000   | 10%        | 3,850,000         | 1,650,000      |
    | 2027 | 6,050,000   | 10%        | 4,235,000         | 1,815,000      |
    | 2028 | 6,655,000   | 10%        | 4,658,500         | 1,996,500      |
    | 2029 | 7,320,500   | 10%        | 5,124,350         | 2,196,150      |
    | 2030 | 8,052,550   | 10%        | 5,636,785         | 2,415,765      |

    Key assumptions:
    - 70% cost ratio maintained throughout
    - 10% compounding growth on prior year revenue
    - No extraordinary items or one-off costs included
""")

FIN_CHAT_ID = "test_financial_chat"
b.chats[FIN_CHAT_ID] = [
    {"role": "user",      "content": "Do a financial analysis and generate 10% planning for next 5 years and export to excel"},
    {"role": "assistant", "content": FINANCIAL_ANALYSIS},
]

try:
    # Verify prompt detection for Excel export
    opts = b._parse_prompt_options(
        "Do a financial analysis and generate 10% planning for next 5 years and export to excel"
    )
    check("Prompt: export_format detected as 'xlsx'",
          opts["export_format"] == "xlsx",
          f"got {opts['export_format']!r}")
    check("Prompt: response_format is 'table' (xlsx forces table)",
          opts["response_format"] == "table",
          f"got {opts['response_format']!r}")

    # Build payload for message index 1 (assistant reply)
    payload = b._get_export_payload(FIN_CHAT_ID, 1)
    check("Payload title not empty",         bool(payload["title"]))

    # Table extraction
    headers, rows = b._extract_markdown_table(FINANCIAL_ANALYSIS)
    check("Table headers extracted",         headers is not None and len(headers) >= 4,
          f"headers={headers}")
    check("Table has 5 data rows (5 years)", rows is not None and len(rows) == 5,
          f"row count={len(rows) if rows else 0}")
    if headers:
        print(f"  → Columns: {headers}")
    if rows:
        for row in rows:
            print(f"    {row}")

    # Write Excel
    xlsx_path = os.path.join(EXPORTS_DIR, "test_financial_export.xlsx")
    b._write_xlsx_export(xlsx_path, payload["rows"], headers, rows)
    check("XLSX file created",  os.path.isfile(xlsx_path),  xlsx_path)
    check("XLSX size > 3 KB",   os.path.getsize(xlsx_path) > 3072,
          f"{os.path.getsize(xlsx_path)} bytes")

    # Verify content with pandas
    import pandas as pd
    df = pd.read_excel(xlsx_path)
    check("Excel has >= 4 columns",  len(df.columns) >= 4,    f"columns={list(df.columns)}")
    check("Excel has 5 data rows",   len(df) == 5,             f"rows={len(df)}")
    print(f"  → Excel preview:\n{df.to_string(index=False)}")
    print(f"  → XLSX saved: {xlsx_path}")

except Exception as exc:
    check("Excel export pipeline", False, traceback.format_exc(limit=2))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"RESULTS: {passed} passed, {failed} failed out of {len(results)} checks")
if failed:
    print("FAILED checks:")
    for name, ok in results:
        if not ok:
            print(f"  ✗ {name}")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
sys.exit(0 if failed == 0 else 1)
