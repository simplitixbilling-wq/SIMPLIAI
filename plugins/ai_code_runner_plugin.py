"""Plugin: AI-callable Python code execution.

Registers /code tool so the AI can autonomously write and execute Python
scripts for calculations, data analysis, visualization, etc.

Runs in a sandboxed subprocess with a timeout. Network and dangerous
OS operations are blocked, but data-science libraries are allowed.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time

PLUGIN_INFO = "AI code runner: /code — lets the AI write & execute Python"

# ── Security ──────────────────────────────────────────────────
_HARD_BLOCKED = re.compile(
    r"""
    subprocess\.\w+|os\.system|os\.popen|os\.exec|os\.spawn|
    shutil\.rmtree|shutil\.move|
    __import__\s*\(\s*['"](?:subprocess|socket|http|urllib|ftplib|
      smtplib|ctypes|webbrowser|antigravity)['"]|
    socket\.\w+|
    requests\.\w+|urllib\.request|
    eval\s*\(|exec\s*\(|compile\s*\(|
    \.remove\s*\(|\.unlink\s*\(|\.rmdir\s*\(
    """,
    re.VERBOSE | re.IGNORECASE
)

_ALLOWED_IMPORTS = {
    "math", "statistics", "random", "re", "json", "csv",
    "datetime", "time", "decimal", "fractions", "collections",
    "itertools", "functools", "operator", "string", "textwrap",
    "pandas", "numpy", "matplotlib", "scipy", "sklearn",
    "seaborn", "plotly",
}

_MAX_CODE_LEN = 15000
_TIMEOUT = 30
_MAX_OUTPUT = 8000


def _extract_code(text: str) -> str:
    """Extract Python code from the args string.

    Handles: /code <raw code>
             /code ```python\n<code>\n```
             /code ```\n<code>\n```
    """
    raw = text.strip()
    if raw.lower().startswith("/code"):
        raw = raw[5:].strip()

    # Strip markdown fences
    fence = re.match(r"^```(?:python)?\s*\n?(.*?)\n?```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)

    return raw.strip()


def _is_safe(code: str) -> tuple:
    """Returns (safe: bool, reason: str)."""
    if not code:
        return False, "Empty code"
    if len(code) > _MAX_CODE_LEN:
        return False, f"Code too long ({len(code)} > {_MAX_CODE_LEN})"
    m = _HARD_BLOCKED.search(code)
    if m:
        return False, f"Blocked operation: {m.group(0).strip()}"
    return True, ""


def _run_code(code: str, work_dir: str) -> dict:
    """Execute code in a subprocess with import gating."""
    # Build the runner script with import whitelist
    runner = f'''
import sys, json, io, importlib, builtins as _builtins, traceback

_ALLOWED = {_ALLOWED_IMPORTS!r}

_original_import = _builtins.__import__
def _gated_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = (name or "").split(".", 1)[0]
    if root not in _ALLOWED and root not in sys.stdlib_module_names:
        raise ImportError(f"Import blocked: {{root}}")
    return _original_import(name, globals, locals, fromlist, level)
_builtins.__import__ = _gated_import

# Redirect matplotlib to non-GUI backend if available
try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass

import os as _os
_os.chdir({work_dir!r})

# Capture output
_stdout = io.StringIO()
_real_stdout = sys.stdout
sys.stdout = _stdout

try:
    exec(compile({code!r}, "<ai_code>", "exec"))
except Exception:
    traceback.print_exc(file=_stdout)
    sys.exit(1)
finally:
    sys.stdout = _real_stdout

output = _stdout.getvalue()
print(output, end="")

# List any files created in work_dir
created = [f for f in _os.listdir({work_dir!r})
           if _os.path.isfile(_os.path.join({work_dir!r}, f))]
if created:
    print(f"\\n[FILES_CREATED] {{json.dumps(created)}}")
'''

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", runner],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            cwd=work_dir,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"Execution timed out after {_TIMEOUT}s"}
    except Exception as e:
        return {"error": f"Failed to start: {e}"}

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    # Truncate large outputs
    if len(stdout) > _MAX_OUTPUT:
        stdout = stdout[:_MAX_OUTPUT] + "\n...[truncated]"
    if len(stderr) > _MAX_OUTPUT:
        stderr = stderr[:_MAX_OUTPUT] + "\n...[truncated]"

    # Extract created files list
    created_files = []
    files_match = re.search(r"\[FILES_CREATED\]\s*(\[.*\])", stdout)
    if files_match:
        try:
            created_files = json.loads(files_match.group(1))
            stdout = stdout[:files_match.start()].strip()
        except json.JSONDecodeError:
            pass

    return {
        "ok": proc.returncode == 0,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": proc.returncode,
        "files": created_files,
    }


def handle_code(app, text: str):
    """Execute Python code provided by the AI or user."""
    code = _extract_code(text)
    safe, reason = _is_safe(code)
    if not safe:
        return {"content": f"⚠ Code blocked: {reason}"}

    # Use agent_temp as working directory
    work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent_temp")
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # If user has uploaded a file, make its extracted text available
    # in agent_temp/uploaded_text.txt (works even for scanned PDFs via OCR)
    uploaded_file = getattr(app, "uploaded_file_path", None) or ""
    uploaded_name = getattr(app, "uploaded_file_name", None) or ""
    uploaded_text_path = ""
    content = getattr(app, "uploaded_content", None)
    if content:
        txt_path = os.path.join(work_dir, "uploaded_text.txt")
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(content)
            uploaded_text_path = txt_path
        except Exception:
            pass

    # Inject file context variables into the code environment
    env_header = (
        f'UPLOADED_FILE = {uploaded_file!r}\n'
        f'UPLOADED_NAME = {uploaded_name!r}\n'
        f'UPLOADED_TEXT = {uploaded_text_path!r}\n'
    )
    code = env_header + code

    result = _run_code(code, work_dir)

    if result.get("error"):
        return {"content": f"❌ {result['error']}"}

    parts = []
    if result.get("stdout"):
        parts.append(f"**Output:**\n```\n{result['stdout']}\n```")
    if result.get("stderr") and not result.get("ok"):
        parts.append(f"**Error:**\n```\n{result['stderr']}\n```")
    if result.get("files"):
        parts.append("**Files created:** " + ", ".join(f"`{f}`" for f in result["files"]))
    if not parts:
        parts.append("✅ Code executed successfully (no output)")

    return {"content": "\n\n".join(parts)}


def register(app):
    app.register_plugin_command(
        "/code",
        handle_code,
        plugin_name="ai_code_runner_plugin",
        description="Execute Python code (data analysis, math, visualization). "
                    "Allowed imports: pandas, numpy, matplotlib, scipy, sklearn, etc. "
                    "Usage: /code <python_code>",
    )
    print("[PLUGIN] ai_code_runner loaded — /code command")


def unregister(app):
    app.unregister_plugin_command("/code")
