"""Settings mixin — app settings panel, plugins UI, toast, first-run wizard (PySide6)."""

import os
import sys

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QComboBox, QScrollArea, QWidget, QFrame,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from app_core.utils import app_data_path


class SettingsMixin:
    """Mixin providing settings-related methods for SimpleAiagentAPP."""

    # ── App settings ───────────────────────────────────────────────

    def _load_app_settings(self):
        self.app_settings = self.chat_db.get_kv("app_settings", {})

    def _save_app_settings(self):
        self.chat_db.set_kv("app_settings", self.app_settings)

    # ── #39 Plugin system UI ───────────────────────────────────────

    def _load_plugins(self):
        try:
            self.plugin_manager.load_all(self)
            count = sum(1 for p in self.plugin_manager.list_plugins() if p["loaded"])
            if count:
                print(f"[PLUGINS] Loaded {count} plugin(s)")
        except Exception as e:
            print(f"[PLUGINS] Error: {e}")

    def _open_plugins_dialog(self):
        dlg = QDialog(self.root)
        dlg.setWindowTitle("🔌 Plugins")
        dlg.resize(450, 400)
        layout = QVBoxLayout(dlg)

        lbl = QLabel("🔌 Installed Plugins")
        lbl.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(lbl)

        folder_lbl = QLabel(f"Folder: {self.plugin_manager.plugins_dir}")
        folder_lbl.setObjectName("Muted")
        layout.addWidget(folder_lbl)

        # Scrollable plugin list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(4, 4, 4, 4)
        list_layout.setSpacing(4)
        list_layout.addStretch()
        scroll.setWidget(list_widget)
        layout.addWidget(scroll, 1)

        def refresh_list():
            while list_layout.count() > 1:
                item = list_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            plugins = self.plugin_manager.list_plugins()
            if not plugins:
                no_lbl = QLabel("No plugins found.\nAdd .py files to plugins/ folder.")
                no_lbl.setObjectName("Muted")
                list_layout.insertWidget(0, no_lbl)
                return

            for p in plugins:
                row = QFrame()
                rlay = QHBoxLayout(row)
                rlay.setContentsMargins(5, 2, 5, 2)
                status = "✅" if p["loaded"] else "⚠️"
                n_lbl = QLabel(f"{status} {p['name']}")
                n_lbl.setFont(QFont("Segoe UI", 13, QFont.Bold))
                rlay.addWidget(n_lbl)
                i_lbl = QLabel(str(p["info"]))
                i_lbl.setObjectName("Muted")
                rlay.addWidget(i_lbl)
                rlay.addStretch()
                count = list_layout.count()
                list_layout.insertWidget(count - 1, row)

        refresh_list()

        btn_row = QHBoxLayout()

        def reload_all():
            self.plugin_manager.loaded.clear()
            self.plugin_manager.load_all(self)
            refresh_list()
            self._show_toast("Plugins reloaded", "info")

        def open_folder():
            os.makedirs(self.plugin_manager.plugins_dir, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(self.plugin_manager.plugins_dir)

        reload_btn = QPushButton("🔄 Reload All")
        reload_btn.setObjectName("GreenBtn")
        reload_btn.clicked.connect(reload_all)
        btn_row.addWidget(reload_btn)

        folder_btn = QPushButton("📂 Open Folder")
        folder_btn.clicked.connect(open_folder)
        btn_row.addWidget(folder_btn)
        layout.addLayout(btn_row)

        dlg.exec()

    # ── App settings panel ─────────────────────────────────────────

    def _open_app_settings_panel(self):
        dlg = QDialog(self.root)
        dlg.setWindowTitle("⚙ App Settings")
        dlg.setFixedSize(440, 400)
        layout = QVBoxLayout(dlg)

        title = QLabel("Global Generation Defaults")
        title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        layout.addWidget(title)

        hint = QLabel("(Per-model settings override temperature and context)")
        hint.setObjectName("Muted")
        layout.addWidget(hint)

        def _make_slider(label_text, value, min_v, max_v, is_float=True):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(170)
            row.addWidget(lbl)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_v, max_v)
            slider.setValue(value)
            row.addWidget(slider, 1)
            if is_float:
                val_lbl = QLabel(f"{value / 100:.2f}")
            else:
                val_lbl = QLabel(str(value))
            val_lbl.setFixedWidth(45)
            row.addWidget(val_lbl)
            if is_float:
                slider.valueChanged.connect(
                    lambda v: val_lbl.setText(f"{v / 100:.2f}"))
            else:
                slider.valueChanged.connect(
                    lambda v: val_lbl.setText(str(v)))
            layout.addLayout(row)
            return slider

        temp_slider = _make_slider(
            "Temperature:",
            int(self.app_settings.get("temperature", 0.25) * 100), 5, 150)
        top_p_slider = _make_slider(
            "Top-p:",
            int(self.app_settings.get("top_p", 0.9) * 100), 10, 100)
        rep_pen_slider = _make_slider(
            "Repeat penalty:",
            int(self.app_settings.get("repeat_penalty", 1.1) * 100), 100, 150)

        # Max response tokens
        resp_row = QHBoxLayout()
        resp_row.addWidget(QLabel("Max response tokens:"))
        resp_combo = QComboBox()
        resp_choices = ["256", "512", "1024", "2048", "4096", "8192", "16384"]
        resp_combo.addItems(resp_choices)
        saved_resp = str(self.app_settings.get("max_response_tokens", "512"))
        if saved_resp in resp_choices:
            resp_combo.setCurrentText(saved_resp)
        else:
            resp_combo.setCurrentText("512")
        resp_row.addWidget(resp_combo)
        layout.addLayout(resp_row)

        layout.addStretch()

        # Buttons
        btn_row = QHBoxLayout()

        def save():
            self.app_settings["temperature"] = round(temp_slider.value() / 100, 3)
            self.app_settings["top_p"] = round(top_p_slider.value() / 100, 3)
            self.app_settings["repeat_penalty"] = round(rep_pen_slider.value() / 100, 3)
            self.app_settings["max_response_tokens"] = int(resp_combo.currentText())
            self._save_app_settings()
            if self.model is not None and hasattr(self, "model_config"):
                self.model_config["temperature"] = self.app_settings["temperature"]
            dlg.accept()
            self._show_toast("App settings saved.", "info")

        def reset():
            self.app_settings = {}
            self._save_app_settings()
            dlg.accept()
            self._show_toast("Settings reset to defaults.", "info")

        save_btn = QPushButton("💾 Save")
        save_btn.setObjectName("GreenBtn")
        save_btn.clicked.connect(save)
        btn_row.addWidget(save_btn, 1)

        reset_btn = QPushButton("↺ Reset defaults")
        reset_btn.clicked.connect(reset)
        btn_row.addWidget(reset_btn, 1)
        layout.addLayout(btn_row)

        dlg.exec()

    # ── GPU Setup dialog ─────────────────────────────────────────

    def _open_gpu_setup(self):
        """Show GPU backend info and provide install commands for
        CUDA / Metal / Vulkan builds of llama-cpp-python."""
        dlg = QDialog(self.root)
        dlg.setWindowTitle("GPU Setup")
        dlg.setFixedSize(520, 400)
        layout = QVBoxLayout(dlg)

        title = QLabel("GPU Backend Configuration")
        title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        layout.addWidget(title)

        # Current detection
        gpu_type = self.gpu_info.get("type", "CPU")
        backend = self.gpu_info.get("backend", "cpu")
        vram = self.gpu_info.get("vram", 0)

        det_text = f"Detected GPU: {gpu_type}"
        if vram:
            det_text += f"  ({vram} GB VRAM)"
        det_lbl = QLabel(det_text)
        det_lbl.setObjectName("Accent")
        layout.addWidget(det_lbl)

        # Check current llama-cpp backend
        llama_backend = self._detect_llama_cpp_backend()
        cur_lbl = QLabel(f"llama-cpp-python backend: {llama_backend}")
        layout.addWidget(cur_lbl)

        # Match check
        if gpu_type != "CPU" and llama_backend == "cpu":
            warn = QLabel(
                f"⚠ Your GPU is {gpu_type} but llama-cpp-python is CPU-only.\n"
                "Reinstall with the correct backend to enable GPU acceleration."
            )
            warn.setStyleSheet("color: #F38BA8; font-weight: bold;")
            warn.setWordWrap(True)
            layout.addWidget(warn)
        elif gpu_type != "CPU" and llama_backend != "cpu":
            ok = QLabel(f"✅ GPU acceleration is active ({llama_backend})")
            ok.setStyleSheet("color: #A6E3A1; font-weight: bold;")
            layout.addWidget(ok)

        layout.addSpacing(10)

        # Install command
        install_cmd, backend_label = self._get_gpu_install_command()

        cmd_title = QLabel(f"Install command for {backend_label}:")
        cmd_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        layout.addWidget(cmd_title)

        from PySide6.QtWidgets import QTextEdit
        cmd_box = QTextEdit()
        cmd_box.setPlainText(install_cmd)
        cmd_box.setReadOnly(True)
        cmd_box.setFixedHeight(60)
        cmd_box.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace; font-size: 11px;"
        )
        layout.addWidget(cmd_box)

        hint = QLabel(
            "Run this command in your terminal/prompt with the virtual environment "
            "activated. Requires CMake and a C++ compiler installed."
        )
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addStretch()

        # Copy + Close
        btn_row = QHBoxLayout()

        def copy_cmd():
            from PySide6.QtWidgets import QApplication
            QApplication.clipboard().setText(install_cmd)
            self._show_toast("Command copied to clipboard.", "info")

        copy_btn = QPushButton("📋 Copy command")
        copy_btn.setObjectName("AccentBtn")
        copy_btn.clicked.connect(copy_cmd)
        btn_row.addWidget(copy_btn, 1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(close_btn, 1)
        layout.addLayout(btn_row)

        dlg.exec()

    # ── Toast notification ─────────────────────────────────────────

    def _show_toast(self, message: str, kind: str = "info", duration: int = 4000):
        colors = {"info": "#3B8ED0", "warn": "#FFC107", "warning": "#FFC107",
                  "error": "#DC3545"}
        bg = colors.get(kind, "#3B8ED0")
        try:
            toast = QLabel(message, self.root)
            toast.setStyleSheet(
                f"background-color: {bg}; color: #FFFFFF; "
                "border-radius: 8px; padding: 10px 14px; font-size: 12px;"
            )
            toast.setWordWrap(True)
            toast.adjustSize()

            rg = self.root.geometry()
            x = rg.width() - toast.width() - 16
            y = rg.height() - toast.height() - 16
            toast.move(max(x, 0), max(y, 0))
            toast.show()
            toast.raise_()

            QTimer.singleShot(duration, toast.deleteLater)
        except Exception:
            pass

    # ── First-run wizard ───────────────────────────────────────────

    def _show_first_run_wizard(self):
        try:
            if getattr(self, "_wizard_shown", False):
                return
            self._wizard_shown = True

            dlg = QDialog(self.root)
            dlg.setWindowTitle("👋 Welcome to SIMPLE_AI")
            dlg.setFixedSize(460, 310)
            layout = QVBoxLayout(dlg)

            welcome = QLabel("👋 Welcome to SIMPLE_AI!")
            welcome.setFont(QFont("Segoe UI", 15, QFont.Bold))
            welcome.setAlignment(Qt.AlignCenter)
            layout.addWidget(welcome)

            info = QLabel(
                "No AI models were found.\n"
                "SIMPLE_AI uses local .gguf model files to run AI completely offline.\n\n"
                "You can download compatible models directly inside the app,\n"
                "or copy any .gguf file into the  models/  folder and restart."
            )
            info.setWordWrap(True)
            info.setAlignment(Qt.AlignCenter)
            layout.addWidget(info)

            models_dir = app_data_path("models")
            dir_lbl = QLabel(f"Models folder:  {models_dir}")
            dir_lbl.setObjectName("Muted")
            dir_lbl.setWordWrap(True)
            layout.addWidget(dir_lbl)

            layout.addStretch()

            btn_row = QHBoxLayout()

            def open_downloader():
                dlg.accept()
                self._open_hf_downloader()

            def open_folder():
                import subprocess as _sp
                _sp.Popen(f'explorer "{models_dir}"')

            dl_btn = QPushButton("⬇ Download a Model")
            dl_btn.setObjectName("AccentBtn")
            dl_btn.clicked.connect(open_downloader)
            btn_row.addWidget(dl_btn, 1)

            folder_btn = QPushButton("📁 Open Models Folder")
            folder_btn.clicked.connect(open_folder)
            btn_row.addWidget(folder_btn, 1)
            layout.addLayout(btn_row)

            dlg.exec()
        except Exception:
            pass
