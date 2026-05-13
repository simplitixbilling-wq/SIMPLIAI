"""
SIMPLE_AI — Local AI Agent (Modular Architecture — PySide6)
═══════════════════════════════════════════════════════════
Thin orchestrator that composes the application from specialised mixins.
Each mixin lives in its own module; standalone helpers are imported directly.
"""

import os
import sys
import threading
import time
from pathlib import Path

# Ensure the project root is on sys.path so shared modules are importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import psutil
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QTimer, Signal, QObject

from styles import get_theme

# ── Standalone modules ──────────────────────────────────────────────
from utils import app_data_path
from database import ChatDatabase, migrate_json_to_sqlite
from plugin_manager import PluginManager
from system_tools import SystemTools, TokenOptimizer  # noqa: F401

# ── Mixin modules ──────────────────────────────────────────────────
from ui_components import UIComponentsMixin
from model_manager import ModelManagerMixin
from chat_manager import ChatManagerMixin
from generation import GenerationMixin
from rag_handler import RAGHandlerMixin
from settings_manager import SettingsMixin


class _Signals(QObject):
    """Thread-safe signal bridge — emit from any thread, slots run on main."""
    update_status = Signal(str)
    add_message = Signal(str, str)          # role, text
    update_textbox = Signal(object, str)    # textbox widget, text
    run_on_main = Signal(object)            # callable


class SimpleAiagentAPP(
    UIComponentsMixin,
    ModelManagerMixin,
    ChatManagerMixin,
    GenerationMixin,
    RAGHandlerMixin,
    SettingsMixin,
):
    def __init__(self):
        # ── 0. Qt APPLICATION ─────────────────────────────────────
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.app.setApplicationName("SIMPLE_AI")
        self.app.setStyle("Fusion")

        self.current_theme = "Dark"
        self.app.setStyleSheet(get_theme(self.current_theme))

        # ── 1. CREATE MAIN WINDOW ─────────────────────────────────
        self.root = QMainWindow()
        self.root.setWindowTitle("SIMPLE_AI Local Agent")

        # Responsive window sizing
        screen = self.app.primaryScreen().availableGeometry()
        sw, sh = screen.width(), screen.height()
        win_w = min(int(sw * 0.88), 1400)
        win_h = min(int(sh * 0.88), 900)
        x = (sw - win_w) // 2
        y = max((sh - win_h) // 2 - 30, 0)
        self.root.setGeometry(x, y, win_w, win_h)
        self.root.setMinimumSize(680, 480)

        if sw >= 1400:
            self.sidebar_w = 280
        elif sw >= 1100:
            self.sidebar_w = 240
        else:
            self.sidebar_w = 200

        # ── Signal bridge for thread-safe UI ──────────────────────
        self.signals = _Signals()
        self.signals.update_status.connect(self._on_update_status)
        self.signals.add_message.connect(self._on_add_message)
        self.signals.update_textbox.connect(self._on_update_textbox)
        self.signals.run_on_main.connect(self._on_run_on_main)

        # ── 2. DATA INIT ──────────────────────────────────────────
        self.chats = {}
        self.current_chat_id = None
        self.chat_counter = 1

        self.model = None
        self.model_path = None
        self.model_map = {}
        self.message_history = []
        self.web_search_enabled = False
        self.generation_in_progress = False
        self.stop_generation_flag = False

        self.rag_manager = None
        self.current_rag_database = None
        self.chat_rag_settings: dict = {}
        self.model_configs: dict = {}
        self.app_settings: dict = {}
        self.chat_system_prompts: dict = {}
        self.attached_image: str = None
        self.voice_listening: bool = False

        # #40 SQLite database
        db_path = app_data_path("chats.db")
        self.chat_db = ChatDatabase(db_path)

        # #41 Config migration
        if migrate_json_to_sqlite(self.chat_db):
            print("[MIGRATE] Imported legacy JSON configs → chats.db")

        # #39 Plugin system
        self.plugin_manager = PluginManager(app_data_path("plugins"))

        # ── 3. SYSTEM DETECTION ────────────────────────────────────
        self.system_ram = round(psutil.virtual_memory().total / (1024**3))
        self.gpu_info = self.detect_gpu_info()

        # ── 4. DYNAMIC CONFIG ─────────────────────────────────────
        gpu_type = self.gpu_info.get("type", "CPU")
        gpu_backend = self.gpu_info.get("backend", "cpu")
        vram = self.gpu_info.get("vram", 0)

        if gpu_type == "NVIDIA":
            if vram >= 4:
                gpu_layers = -1
            elif vram >= 2:
                gpu_layers = 20
            else:
                gpu_layers = 8

            self.config = {
                "mode": "GPU",
                "n_gpu_layers": gpu_layers,
                "n_threads": 4,
                "max_tokens": 2048
            }
        elif gpu_type == "APPLE_METAL":
            # Metal: unified memory, offload all layers
            self.config = {
                "mode": "GPU (Metal)",
                "n_gpu_layers": -1,
                "n_threads": max(4, (os.cpu_count() or 8) - 2),
                "max_tokens": 2048
            }
        elif gpu_type == "AMD":
            # Vulkan: offload layers based on detected VRAM
            if vram >= 4:
                gpu_layers = -1
            elif vram >= 2:
                gpu_layers = 20
            else:
                gpu_layers = 8

            self.config = {
                "mode": "GPU (Vulkan)",
                "n_gpu_layers": gpu_layers,
                "n_threads": max(4, (os.cpu_count() or 8) - 2),
                "max_tokens": 2048
            }
        else:
            if self.system_ram <= 8:
                self.config = {
                    "mode": "LOW_RAM",
                    "n_gpu_layers": 0,
                    "n_threads": 4,
                    "max_tokens": 800
                }
            elif self.system_ram <= 16:
                self.config = {
                    "mode": "CPU",
                    "n_gpu_layers": 0,
                    "n_threads": max(8, os.cpu_count() - 2) if os.cpu_count() else 8,
                    "max_tokens": 1024
                }
            else:
                self.config = {
                    "mode": "CPU_HIGH",
                    "n_gpu_layers": 0,
                    "n_threads": max(10, os.cpu_count() - 2) if os.cpu_count() else 10,
                    "max_tokens": 2048
                }

        # ── 5. BUILD UI ───────────────────────────────────────────
        self._setup_ui()

        # ── 6. LOAD DATA ─────────────────────────────────────────
        self._scan_models_on_startup()
        self._load_model_configs()
        self._load_app_settings()
        self._load_saved_chats()
        QTimer.singleShot(100, self._init_rag_async)
        QTimer.singleShot(300, self._load_plugins)
        self._start_auto_save_timer()

        if not self.chats:
            self.new_chat()

        # ── 7. START MONITOR ──────────────────────────────────────
        self.start_system_monitor()

        # ── 8. STATUS ─────────────────────────────────────────────
        self.update_status(
            f"{self.config['mode']} | RAM: {self.system_ram}GB | VRAM: {self.gpu_info.get('vram', 0)}GB"
        )

        self._shutting_down = False
        self.root.show()

    # ── Signal slots (thread-safe UI updates) ─────────────────────

    def _on_update_status(self, text):
        self.update_status(text)

    def _on_add_message(self, role, text):
        self.add_message(role, text)

    def _on_update_textbox(self, textbox, text):
        self._set_textbox_text(textbox, text)

    def _on_run_on_main(self, func):
        try:
            func()
        except Exception:
            pass

    def _run_on_main(self, func):
        """Schedule func on the main thread. Safe to call from any thread."""
        if threading.current_thread() is threading.main_thread():
            func()
        else:
            self.signals.run_on_main.emit(func)

    # ── Compatibility shim: root.after → QTimer ──────────────────

    def _after(self, ms, callback):
        """Schedule callback on main thread after ms milliseconds."""
        QTimer.singleShot(ms, callback)

    def _start_auto_save_timer(self):
        self._auto_save_qt = QTimer()
        self._auto_save_qt.timeout.connect(self._auto_save_timer_tick)
        self._auto_save_qt.start(60000)

    def run(self):
        self.app.aboutToQuit.connect(self._on_shutdown)
        sys.exit(self.app.exec())

    def _on_shutdown(self):
        self._shutting_down = True


if __name__ == "__main__":
    app = SimpleAiagentAPP()
    app.run()

