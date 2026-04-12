"""SIMPLE_AI — pywebview entry point.

Run this instead of agent.py to launch the web-based UI
while keeping the PySide6 version untouched.

Usage:
    python agent_web.py
"""

import os
import sys

# Determine the application root directory
# When frozen by PyInstaller (onedir), sys.executable is the exe path
# and __file__ may point inside _internal/. Use the exe's parent instead.
if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(APP_DIR)

import webview
from webview.errors import JavascriptException
from bridge import Bridge
from local_api_server import LocalApiServer
from utils import resource_path


def _install_pywebview_callback_race_guard():
    """Ignore stale pywebview return-callback races after page reload/dispose.

    pywebview can raise JavascriptException when an async Python API call
    resolves after the JS callback registry was reset (for example after a
    reload). Those errors are benign but noisy; keep all other JS exceptions.
    """
    try:
        original_evaluate_js = webview.window.Window.evaluate_js
    except Exception:
        return

    def safe_evaluate_js(self, *args, **kwargs):
        try:
            return original_evaluate_js(self, *args, **kwargs)
        except JavascriptException as e:
            msg = str(e)
            if "_returnValuesCallbacks" in msg and "is not a function" in msg:
                return None
            raise

    webview.window.Window.evaluate_js = safe_evaluate_js


def main():
    _install_pywebview_callback_race_guard()
    bridge = Bridge()

    # Start secure localhost API for external integrations (Excel/Docs/scripts).
    api_server = LocalApiServer(bridge)
    api_meta = api_server.start()
    print(f"[LOCAL API] http://{api_meta['host']}:{api_meta['port']}")
    print(f"[LOCAL API] X-API-Key: {api_meta['api_key']}")
    print(f"[LOCAL API] API key file: {api_meta['api_key_file']}")

    # web/ is a read-only bundled resource (--add-data puts it in _MEIPASS)
    web_dir = resource_path("web")
    index_url = os.path.join(web_dir, "index.html")
    # webview_data is writable user data — lives next to the exe
    storage_dir = os.path.join(APP_DIR, "webview_data")
    os.makedirs(storage_dir, exist_ok=True)

    window = webview.create_window(
        title="SIMPLE_AI",
        url=index_url,
        js_api=bridge,
        width=1200,
        height=800,
        min_size=(900, 600),
        text_select=True,
    )

    try:
        window.events.closed += lambda: api_server.stop()
    except Exception:
        pass

    webview.start(
        debug=("--debug" in sys.argv),
        private_mode=False,
        storage_path=storage_dir,
    )


if __name__ == "__main__":
    main()
