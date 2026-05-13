"""UI components mixin — setup_ui, message rendering, markdown, compare, voice, image (PySide6)."""

import gc
import os
import re
import threading
import time
import tempfile
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QLabel, QPushButton, QLineEdit, QTextEdit,
    QComboBox, QProgressBar, QScrollArea, QSplitter,
    QSizePolicy, QFileDialog, QDialog, QRadioButton,
    QButtonGroup, QApplication,
)
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import (
    QFont, QTextCharFormat, QColor, QSyntaxHighlighter,
    QTextCursor, QShortcut, QKeySequence, QIcon,
)

from styles import COLORS

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


class UIComponentsMixin:
    """Mixin providing UI construction and rendering methods for SimpleAiagentAPP."""

    # ── Main UI build ──────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.root.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ════════════════════ TOP BAR ════════════════════
        self.top_bar = QFrame()
        self.top_bar.setObjectName("TopBar")
        self.top_bar.setFixedHeight(44)
        tb = QHBoxLayout(self.top_bar)
        tb.setContentsMargins(10, 0, 12, 0)
        tb.setSpacing(10)

        self.menu_toggle = QPushButton("☰")
        self.menu_toggle.setObjectName("IconBtn")
        self.menu_toggle.setFixedSize(34, 34)
        self.menu_toggle.clicked.connect(self.toggle_sidebar)
        tb.addWidget(self.menu_toggle)

        self.title_label = QLabel("SIMPLE_AI")
        self.title_label.setObjectName("BrandLabel")
        self.title_label.setFont(QFont("Segoe UI", 15, QFont.Bold))
        tb.addWidget(self.title_label)

        tb.addStretch()

        # Model selector in top bar
        self.model_menu = QComboBox()
        self.model_menu.addItem("No model found")
        self.model_menu.setMinimumWidth(160)
        self.model_menu.setMaximumWidth(240)
        self.model_menu.currentTextChanged.connect(self._on_model_select)
        tb.addWidget(self.model_menu)

        # Tool icons after model selector
        per_model_btn = QPushButton("\u2699")
        per_model_btn.setObjectName("IconBtn")
        per_model_btn.setFixedSize(30, 30)
        per_model_btn.setToolTip("Per-model settings")
        per_model_btn.clicked.connect(self._open_per_model_settings)
        tb.addWidget(per_model_btn)

        hf_btn = QPushButton("\u2193")
        hf_btn.setObjectName("IconBtn")
        hf_btn.setFixedSize(30, 30)
        hf_btn.setToolTip("Download models")
        hf_btn.clicked.connect(self._open_hf_downloader)
        tb.addWidget(hf_btn)

        sys_prompt_btn = QPushButton("\u270E")
        sys_prompt_btn.setObjectName("IconBtn")
        sys_prompt_btn.setFixedSize(30, 30)
        sys_prompt_btn.setToolTip("System prompt")
        sys_prompt_btn.clicked.connect(self._open_system_prompt_dialog)
        tb.addWidget(sys_prompt_btn)

        root_layout.addWidget(self.top_bar)

        # ════════════════════ BODY ════════════════════
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # ──────────── SIDEBAR ────────────
        self.sidebar_visible = True
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(self.sidebar_w)
        sb = QVBoxLayout(self.sidebar)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(0)

        # ── Model controls section ──
        model_sec = QWidget()
        model_sec.setObjectName("SidebarSection")
        ms = QVBoxLayout(model_sec)
        ms.setContentsMargins(12, 10, 12, 8)
        ms.setSpacing(6)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.load_button = QPushButton("Load Model")
        self.load_button.setObjectName("AccentBtn")
        self.load_button.setFixedHeight(30)
        self.load_button.clicked.connect(self.load_model)
        btn_row.addWidget(self.load_button)

        self.new_chat_button = QPushButton("+ New Chat")
        self.new_chat_button.setObjectName("GhostBtn")
        self.new_chat_button.setFixedHeight(30)
        self.new_chat_button.clicked.connect(self.new_chat)
        btn_row.addWidget(self.new_chat_button)
        ms.addLayout(btn_row)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        ms.addWidget(self.progress_bar)

        # Status labels
        self.loaded_model_label = QLabel("No model loaded")
        self.loaded_model_label.setObjectName("StatusLabel")
        self.loaded_model_label.setWordWrap(True)
        ms.addWidget(self.loaded_model_label)

        self.status_indicator = QLabel("Ready")
        self.status_indicator.setObjectName("Accent")
        self.status_indicator.setWordWrap(True)
        ms.addWidget(self.status_indicator)

        self.system_monitor_label = QLabel("")
        self.system_monitor_label.setObjectName("Muted")

        # Context window bar
        ctx_row = QHBoxLayout()
        ctx_row.setSpacing(4)
        self.ctx_bar = QProgressBar()
        self.ctx_bar.setRange(0, 100)
        self.ctx_bar.setValue(0)
        self.ctx_bar.setTextVisible(False)
        self.ctx_bar.setFixedHeight(3)
        ctx_row.addWidget(self.ctx_bar, 1)
        self.ctx_label = QLabel("")
        self.ctx_label.setObjectName("Muted")
        self.ctx_label.setFixedWidth(42)
        ctx_row.addWidget(self.ctx_label)
        ms.addLayout(ctx_row)

        sb.addWidget(model_sec)

        # ── Chats section ──
        chats_hdr = QWidget()
        chats_hdr.setObjectName("SectionHeader")
        ch = QHBoxLayout(chats_hdr)
        ch.setContentsMargins(12, 10, 8, 4)
        ch.setSpacing(4)

        self.chat_list_label = QLabel("CHATS")
        self.chat_list_label.setObjectName("SectionTitle")
        ch.addWidget(self.chat_list_label)
        ch.addStretch()

        sb.addWidget(chats_hdr)

        # Chat search
        search_wrap = QWidget()
        sw = QHBoxLayout(search_wrap)
        sw.setContentsMargins(10, 2, 10, 4)
        self.chat_search_box = QLineEdit()
        self.chat_search_box.setPlaceholderText("Search chats...")
        self.chat_search_box.setFixedHeight(28)
        self.chat_search_box.textChanged.connect(self._filter_chat_list)
        sw.addWidget(self.chat_search_box)
        sb.addWidget(search_wrap)

        # Chat list (scrollable)
        self.chat_list_scroll = QScrollArea()
        self.chat_list_scroll.setWidgetResizable(True)
        self.chat_list_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_list_widget = QWidget()
        self.chat_list_layout = QVBoxLayout(self.chat_list_widget)
        self.chat_list_layout.setContentsMargins(8, 2, 8, 2)
        self.chat_list_layout.setSpacing(2)
        self.chat_list_layout.addStretch()
        self.chat_list_scroll.setWidget(self.chat_list_widget)
        sb.addWidget(self.chat_list_scroll, 1)

        # ── Separator between Chats and Knowledge ──
        self.chat_rag_sep = QFrame()
        self.chat_rag_sep.setFrameShape(QFrame.HLine)
        self.chat_rag_sep.setStyleSheet(self._separator_style(with_margin=True))
        sb.addWidget(self.chat_rag_sep)

        # ── Knowledge / RAG section ──
        rag_hdr = QWidget()
        rag_hdr.setObjectName("SectionHeader")
        rh = QHBoxLayout(rag_hdr)
        rh.setContentsMargins(12, 6, 8, 4)
        rh.setSpacing(4)

        self.rag_label = QLabel("KNOWLEDGE")
        self.rag_label.setObjectName("SectionTitle")
        rh.addWidget(self.rag_label)
        rh.addStretch()

        self.add_rag_button = QPushButton("+")
        self.add_rag_button.setObjectName("TinyBtn")
        self.add_rag_button.setFixedSize(22, 22)
        self.add_rag_button.setToolTip("Create RAG database")
        self.add_rag_button.clicked.connect(self._open_rag_create_dialog)
        rh.addWidget(self.add_rag_button)

        sb.addWidget(rag_hdr)

        # RAG list (scrollable)
        self.rag_list_scroll = QScrollArea()
        self.rag_list_scroll.setWidgetResizable(True)
        self.rag_list_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.rag_list_scroll.setMaximumHeight(120)
        self.rag_list_widget = QWidget()
        self.rag_list_layout = QVBoxLayout(self.rag_list_widget)
        self.rag_list_layout.setContentsMargins(8, 2, 8, 4)
        self.rag_list_layout.setSpacing(2)
        self.rag_list_layout.addStretch()
        self.rag_list_scroll.setWidget(self.rag_list_widget)
        sb.addWidget(self.rag_list_scroll)

        self.rag_widgets = {}

        # ── Bottom settings area ──
        sb.addStretch()

        self.sidebar_bottom_sep = QFrame()
        self.sidebar_bottom_sep.setFrameShape(QFrame.HLine)
        self.sidebar_bottom_sep.setStyleSheet(self._separator_style())
        sb.addWidget(self.sidebar_bottom_sep)

        settings_area = QWidget()
        sa_layout = QVBoxLayout(settings_area)
        sa_layout.setContentsMargins(10, 8, 10, 10)
        sa_layout.setSpacing(4)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(6)

        self.compare_btn = QPushButton("Compare Models")
        self.compare_btn.setObjectName("GhostBtn")
        self.compare_btn.setFixedHeight(28)
        self.compare_btn.clicked.connect(self._open_compare_dialog)
        bottom_row.addWidget(self.compare_btn)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setObjectName("GhostBtn")
        self.settings_btn.setFixedHeight(28)
        self.settings_btn.clicked.connect(self._toggle_settings_dropdown)
        bottom_row.addWidget(self.settings_btn)

        sa_layout.addLayout(bottom_row)
        sa_layout.addWidget(self.system_monitor_label)
        sb.addWidget(settings_area)

        body_layout.addWidget(self.sidebar)

        # ──────────── CHAT AREA ────────────
        chat_panel = QWidget()
        chat_panel.setObjectName("ChatArea")
        cp = QVBoxLayout(chat_panel)
        cp.setContentsMargins(0, 0, 0, 0)
        cp.setSpacing(0)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setObjectName("ChatScroll")
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.chat_frame = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_frame)
        self.chat_layout.setContentsMargins(48, 24, 48, 24)
        self.chat_layout.setSpacing(16)
        self.chat_layout.addStretch()
        self.chat_scroll.setWidget(self.chat_frame)
        cp.addWidget(self.chat_scroll, 1)

        # ──────────── INPUT BAR ────────────
        self.input_container = QFrame()
        self.input_container.setObjectName("InputBar")
        ic = QVBoxLayout(self.input_container)
        ic.setContentsMargins(48, 8, 48, 12)
        ic.setSpacing(0)

        input_row = QHBoxLayout()
        input_row.setSpacing(6)

        # Left action icons
        self.upload_button = QPushButton("⇪")
        self.upload_button.setObjectName("InputIcon")
        self.upload_button.setFixedSize(34, 34)
        self.upload_button.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.upload_button.setToolTip("Upload file")
        self.upload_button.clicked.connect(self.upload_file)
        input_row.addWidget(self.upload_button)

        self.web_search_button = QPushButton("◎")
        self.web_search_button.setObjectName("InputIcon")
        self.web_search_button.setFixedSize(34, 34)
        self.web_search_button.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.web_search_button.setToolTip("Web search")
        self.web_search_button.clicked.connect(self.toggle_web_search)
        input_row.addWidget(self.web_search_button)

        self.image_attach_btn = QPushButton("▦")
        self.image_attach_btn.setObjectName("InputIcon")
        self.image_attach_btn.setFixedSize(34, 34)
        self.image_attach_btn.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.image_attach_btn.setToolTip("Attach image")
        self.image_attach_btn.clicked.connect(self._attach_image)
        input_row.addWidget(self.image_attach_btn)

        self.voice_btn = QPushButton("◉")
        self.voice_btn.setObjectName("InputIcon")
        self.voice_btn.setFixedSize(34, 34)
        self.voice_btn.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.voice_btn.setToolTip("Voice input")
        self.voice_btn.clicked.connect(self._toggle_voice_input)
        input_row.addWidget(self.voice_btn)

        # Input field
        self.input_box = QLineEdit()
        self.input_box.setObjectName("ChatInput")
        self.input_box.setPlaceholderText("Ask anything...")
        self.input_box.setMinimumHeight(38)
        self.input_box.setFont(QFont("Segoe UI", 13))
        self.input_box.returnPressed.connect(self.on_send)
        self.input_box.textChanged.connect(self._update_token_counter)
        input_row.addWidget(self.input_box, 1)

        self.token_counter_label = QLabel("")
        self.token_counter_label.setObjectName("Muted")
        self.token_counter_label.setFixedWidth(30)
        input_row.addWidget(self.token_counter_label)

        # Stop & Send buttons
        self.stop_btn = QPushButton("■")
        self.stop_btn.setObjectName("StopBtn")
        self.stop_btn.setFixedSize(34, 34)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("Stop generation")
        self.stop_btn.clicked.connect(self.stop_generation)
        input_row.addWidget(self.stop_btn)

        self.send_btn = QPushButton("➤")
        self.send_btn.setObjectName("SendBtn")
        self.send_btn.setFixedSize(38, 38)
        self.send_btn.setEnabled(False)
        self.send_btn.setToolTip("Send message")
        self.send_btn.clicked.connect(self.on_send)
        input_row.addWidget(self.send_btn)

        ic.addLayout(input_row)

        cp.addWidget(self.input_container)
        body_layout.addWidget(chat_panel, 1)
        root_layout.addWidget(body, 1)

        # ── Keyboard shortcuts ──
        QShortcut(QKeySequence("Ctrl+N"), self.root, activated=self.new_chat)
        QShortcut(QKeySequence("Ctrl+L"), self.root, activated=self.load_model)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self.root, activated=self._open_app_settings_panel)
        QShortcut(QKeySequence("Ctrl+E"), self.root, activated=lambda: self._export_chat() if hasattr(self, '_export_chat') else None)
        QShortcut(QKeySequence("Escape"), self.root, activated=lambda: self.stop_generation() if self.generation_in_progress else None)
        QShortcut(QKeySequence("Ctrl+B"), self.root, activated=self.toggle_sidebar)

        # ── Tracking attributes ──
        self.typing_indicator = None
        self._last_textbox = None
        self._last_container = None

    # ── UI helpers ──────────────────────────────────────────────────────

    def _safe_destroy_children(self, layout_or_widget):
        """Remove all child widgets from a layout or widget."""
        if isinstance(layout_or_widget, QWidget):
            layout = layout_or_widget.layout()
        else:
            layout = layout_or_widget
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
            elif item.layout():
                self._safe_destroy_children(item.layout())

    def toggle_sidebar(self):
        if self.sidebar_visible:
            self.sidebar.hide()
            self.sidebar_visible = False
        else:
            self.sidebar.show()
            self.sidebar_visible = True

    def _toggle_settings_dropdown(self):
        """Unified settings dialog: Theme, App Settings, Plugins."""
        from styles import get_theme
        dlg = QDialog(self.root)
        dlg.setWindowTitle("Settings")
        dlg.setFixedSize(260, 300)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(12)

        # ── Theme section ──
        lbl = QLabel("Theme")
        lbl.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(lbl)

        grp = QButtonGroup(dlg)
        rb_dark = QRadioButton("Dark")
        rb_light = QRadioButton("Light")
        grp.addButton(rb_dark)
        grp.addButton(rb_light)
        if self.current_theme == "Dark":
            rb_dark.setChecked(True)
        else:
            rb_light.setChecked(True)
        theme_row = QHBoxLayout()
        theme_row.setSpacing(16)
        theme_row.addWidget(rb_dark)
        theme_row.addWidget(rb_light)
        theme_row.addStretch()
        layout.addLayout(theme_row)

        def apply_theme():
            self.current_theme = "Dark" if rb_dark.isChecked() else "Light"
            self.app.setStyleSheet(get_theme(self.current_theme))
            self._refresh_theme_ui()

        apply_btn = QPushButton("Apply Theme")
        apply_btn.setObjectName("AccentBtn")
        apply_btn.setFixedHeight(30)
        apply_btn.clicked.connect(apply_theme)
        layout.addWidget(apply_btn)

        # ── Separator ──
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(self._separator_style())
        layout.addWidget(sep)

        # ── App Settings ──
        app_settings_btn = QPushButton("App Settings")
        app_settings_btn.setObjectName("GhostBtn")
        app_settings_btn.setFixedHeight(32)
        app_settings_btn.clicked.connect(lambda: (dlg.accept(), self._open_app_settings_panel()))
        layout.addWidget(app_settings_btn)

        # ── Plugins ──
        plugins_btn = QPushButton("Plugins")
        plugins_btn.setObjectName("GhostBtn")
        plugins_btn.setFixedHeight(32)
        plugins_btn.clicked.connect(lambda: (dlg.accept(), self._open_plugins_dialog()))
        layout.addWidget(plugins_btn)

        # ── GPU Setup ──
        gpu_type = self.gpu_info.get("type", "CPU")
        gpu_label = "GPU Setup"
        if gpu_type == "NVIDIA":
            gpu_label = "GPU Setup (NVIDIA)"
        elif gpu_type == "APPLE_METAL":
            gpu_label = "GPU Setup (Metal)"
        elif gpu_type == "AMD":
            gpu_label = "GPU Setup (Vulkan)"
        gpu_btn = QPushButton(gpu_label)
        gpu_btn.setObjectName("GhostBtn")
        gpu_btn.setFixedHeight(32)
        gpu_btn.clicked.connect(lambda: (dlg.accept(), self._open_gpu_setup()))
        layout.addWidget(gpu_btn)

        layout.addStretch()

        # ── Close ──
        close_btn = QPushButton("Close")
        close_btn.setObjectName("GhostBtn")
        close_btn.setFixedHeight(30)
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)

        dlg.exec()

    def _separator_style(self, with_margin=False):
        color = "#1c1f2e" if self.current_theme == "Dark" else "#d6dae4"
        margin = " margin: 4px 12px;" if with_margin else ""
        return f"background-color: {color}; max-height: 1px;{margin}"

    def _refresh_theme_ui(self):
        if hasattr(self, "chat_rag_sep") and self.chat_rag_sep is not None:
            self.chat_rag_sep.setStyleSheet(self._separator_style(with_margin=True))
        if hasattr(self, "sidebar_bottom_sep") and self.sidebar_bottom_sep is not None:
            self.sidebar_bottom_sep.setStyleSheet(self._separator_style())
        if getattr(self, "current_chat_id", None) and getattr(self, "message_history", None) is not None:
            self._load_chat(self.current_chat_id)
        else:
            self._clear_chat_area()

    def _on_model_select(self, choice):
        self.model_path = self.model_map.get(choice)

    def update_status(self, text):
        self.status_indicator.setText(text)

    def _scroll_chat(self, position="bottom"):
        """Scroll chat area to top or bottom."""
        def do_scroll():
            sb = self.chat_scroll.verticalScrollBar()
            if position == "bottom":
                sb.setValue(sb.maximum())
            else:
                sb.setValue(0)
        QTimer.singleShot(50, do_scroll)

    def _update_ctx_bar(self, used_tokens: int = 0, total_tokens: int = 0):
        """Update context window visualizer."""
        try:
            if total_tokens <= 0:
                self.ctx_bar.setValue(0)
                self.ctx_label.setText("")
                return
            ratio = min(used_tokens / total_tokens, 1.0)
            pct = int(ratio * 100)
            self.ctx_bar.setValue(pct)

            if pct < 60:
                col = "#4ade80" if self.current_theme == "Dark" else "#16a34a"
            elif pct < 85:
                col = "#fbbf24" if self.current_theme == "Dark" else "#d97706"
            else:
                col = "#f87171" if self.current_theme == "Dark" else "#dc2626"

            self.ctx_bar.setStyleSheet(
                f"QProgressBar::chunk {{ background-color: {col}; border-radius: 2px; }}"
            )
            self.ctx_label.setText(f"{pct}%")
        except Exception:
            pass

    # ── Multi-model Compare ─────────────────────────────────────

    def _open_compare_dialog(self):
        """Compare responses from two models side-by-side."""
        if not self.model_map or len(self.model_map) < 2:
            self._show_toast("Need at least 2 models in models/ folder", "warning")
            return

        dlg = QDialog(self.root)
        dlg.setWindowTitle("Multi-Model Compare")
        dlg.resize(900, 650)
        layout = QVBoxLayout(dlg)

        model_names = list(self.model_map.keys())

        # Model selectors
        top = QHBoxLayout()
        col_a = QVBoxLayout()
        col_a.addWidget(QLabel("Model A:"))
        model_a_combo = QComboBox()
        model_a_combo.addItems(model_names)
        col_a.addWidget(model_a_combo)
        top.addLayout(col_a)

        col_b = QVBoxLayout()
        col_b.addWidget(QLabel("Model B:"))
        model_b_combo = QComboBox()
        model_b_combo.addItems(model_names)
        if len(model_names) > 1:
            model_b_combo.setCurrentIndex(1)
        col_b.addWidget(model_b_combo)
        top.addLayout(col_b)
        layout.addLayout(top)

        # Prompt
        layout.addWidget(QLabel("Prompt:"))
        prompt_entry = QLineEdit()
        prompt_entry.setPlaceholderText("Enter a prompt to compare...")
        prompt_entry.setMinimumHeight(40)
        layout.addWidget(prompt_entry)

        # Results
        results_top = QHBoxLayout()
        lbl_a = QLabel("Model A Response:")
        lbl_a.setObjectName("Accent")
        lbl_a.setFont(QFont("Segoe UI", 12, QFont.Bold))
        results_top.addWidget(lbl_a)
        lbl_b = QLabel("Model B Response:")
        lbl_b.setFont(QFont("Segoe UI", 12, QFont.Bold))
        lbl_b.setStyleSheet("color: #f87171;" if self.current_theme == "Dark" else "color: #dc2626;")
        results_top.addWidget(lbl_b)
        layout.addLayout(results_top)

        text_row = QHBoxLayout()
        text_a = QTextEdit()
        text_a.setReadOnly(True)
        text_row.addWidget(text_a)
        text_b = QTextEdit()
        text_b.setReadOnly(True)
        text_row.addWidget(text_b)
        layout.addLayout(text_row, 1)

        # Bottom
        bottom = QHBoxLayout()
        status_lbl = QLabel("Ready")
        status_lbl.setObjectName("Muted")
        bottom.addWidget(status_lbl)
        bottom.addStretch()

        def run_compare():
            prompt_text = prompt_entry.text().strip()
            if not prompt_text:
                self._show_toast("Enter a prompt first", "warning")
                return
            compare_btn.setEnabled(False)
            text_a.clear()
            text_b.clear()
            text_a.setPlainText("Loading model A...")
            text_b.setPlainText("Waiting...")

            def compare_thread():
                name_a = model_a_combo.currentText()
                name_b = model_b_combo.currentText()
                path_a = self.model_map.get(name_a)
                path_b = self.model_map.get(name_b)

                for label, path, textbox in [("A", path_a, text_a), ("B", path_b, text_b)]:
                    QTimer.singleShot(0, lambda l=label: status_lbl.setText(f"Loading Model {l}..."))
                    QTimer.singleShot(0, lambda tb=textbox, l=label: tb.setPlainText(f"Loading Model {l}..."))
                    try:
                        temp_model = Llama(
                            model_path=path,
                            n_threads=self.config["n_threads"],
                            n_gpu_layers=self.config["n_gpu_layers"],
                            n_ctx=2048, verbose=False
                        )
                        QTimer.singleShot(0, lambda l=label: status_lbl.setText(f"Generating with Model {l}..."))
                        QTimer.singleShot(0, lambda tb=textbox: tb.clear())

                        out = temp_model(
                            f"Question: {prompt_text}\n\nAnswer:",
                            max_tokens=512, temperature=0.25, top_p=0.9, stream=False
                        )
                        resp = out["choices"][0]["text"].strip() if out.get("choices") else "No response"
                        QTimer.singleShot(0, lambda tb=textbox, r=resp: tb.setPlainText(r))
                        del temp_model
                        gc.collect()
                    except Exception as e:
                        QTimer.singleShot(0, lambda tb=textbox, err=str(e): tb.setPlainText(f"Error: {err}"))

                QTimer.singleShot(0, lambda: status_lbl.setText("Comparison complete"))
                QTimer.singleShot(0, lambda: compare_btn.setEnabled(True))

                if self.model_path:
                    QTimer.singleShot(0, lambda: status_lbl.setText("Reloading original model..."))
                    try:
                        self.model = Llama(
                            model_path=self.model_path,
                            n_threads=self.config["n_threads"],
                            n_gpu_layers=self.config["n_gpu_layers"],
                            n_ctx=getattr(self, 'actual_n_ctx', 2048),
                            verbose=False
                        )
                        QTimer.singleShot(0, lambda: status_lbl.setText("Done — original model restored"))
                    except Exception:
                        QTimer.singleShot(0, lambda: status_lbl.setText("Could not reload original model"))

            threading.Thread(target=compare_thread, daemon=True).start()

        compare_btn = QPushButton("Compare")
        compare_btn.setObjectName("AccentBtn")
        compare_btn.setFixedWidth(120)
        compare_btn.clicked.connect(run_compare)
        bottom.addWidget(compare_btn)
        layout.addLayout(bottom)

        dlg.exec()

    # ── Image Input (Multimodal) ────────────────────────────────

    def _attach_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self.root, "Attach Image", "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp *.webp)"
        )
        if path and os.path.isfile(path):
            self.attached_image = path
            fname = os.path.basename(path)
            self.image_attach_btn.setText("✓")
            self.image_attach_btn.setStyleSheet(
                "QPushButton { background-color: #22c55e; color: #ffffff; border: none; border-radius: 17px; }"
            )
            self._show_toast(f"Image attached: {fname}", "info")
        else:
            self.attached_image = None
            self.image_attach_btn.setText("▦")
            self.image_attach_btn.setStyleSheet("")

    def _get_image_base64(self, path: str) -> str:
        import base64
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _extract_text_from_image(self, path: str) -> str:
        try:
            from rapidocr_onnxruntime import RapidOCR
            from PIL import Image, ImageOps, ImageEnhance, ImageFilter
        except Exception:
            return ""

        try:
            engine = RapidOCR()

            def _collect_lines(ocr_result):
                lines = []
                for item in ocr_result or []:
                    if len(item) >= 2 and isinstance(item[1], str):
                        text = item[1].strip()
                        if not text:
                            continue
                        confidence = 0.0
                        if len(item) >= 3:
                            try:
                                confidence = float(item[2])
                            except Exception:
                                confidence = 0.0
                        lines.append((confidence, text))
                return lines

            best_text = ""
            best_score = -1.0

            # Pass 1: original image
            result, _elapsed = engine(path)
            base_lines = _collect_lines(result)
            if base_lines:
                score = sum(conf for conf, _txt in base_lines)
                text = "\n".join(txt for _conf, txt in base_lines)
                if score > best_score and text.strip():
                    best_text, best_score = text, score

            # Pass 2: enhanced variants for difficult screenshots/photos
            with tempfile.TemporaryDirectory() as temp_dir:
                src = Image.open(path).convert("L")
                variants = []

                # Upscaled + autocontrast
                up = src.resize((src.width * 2, src.height * 2))
                up = ImageOps.autocontrast(up)
                variants.append(up)

                # High-contrast sharpened grayscale
                sharp = ImageEnhance.Contrast(src).enhance(2.2)
                sharp = sharp.filter(ImageFilter.SHARPEN)
                variants.append(sharp)

                # Binary threshold variant
                binary = src.point(lambda p: 255 if p > 160 else 0, mode="1").convert("L")
                variants.append(binary)

                for idx, image in enumerate(variants):
                    candidate_path = os.path.join(temp_dir, f"ocr_variant_{idx}.png")
                    image.save(candidate_path)
                    cand_result, _cand_elapsed = engine(candidate_path)
                    cand_lines = _collect_lines(cand_result)
                    if not cand_lines:
                        continue
                    cand_score = sum(conf for conf, _txt in cand_lines)
                    cand_text = "\n".join(txt for _conf, txt in cand_lines)
                    if cand_score > best_score and cand_text.strip():
                        best_text, best_score = cand_text, cand_score

            return best_text.strip()
        except Exception:
            return ""

    def _try_multimodal_generate(self, prompt_text: str, image_path: str) -> str:
        try:
            from llama_cpp.llama_chat_format import Llava15ChatHandler
            clip_path = None
            model_dir = os.path.dirname(self.model_path)
            for f in os.listdir(model_dir):
                if "mmproj" in f.lower() or "clip" in f.lower():
                    clip_path = os.path.join(model_dir, f)
                    break

            if not clip_path:
                return None

            chat_handler = Llava15ChatHandler(clip_model_path=clip_path)
            import base64
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            ext = os.path.splitext(image_path)[1].lower().lstrip(".")
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                    "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp"}.get(ext, "image/png")

            result = self.model.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                        {"type": "text", "text": prompt_text}
                    ]
                }],
                max_tokens=512,
                chat_handler=chat_handler
            )
            return result["choices"][0]["message"]["content"]
        except Exception:
            return None

    # ── Voice Input / Output ────────────────────────────────────

    def _toggle_voice_input(self):
        if self.voice_listening:
            self.voice_listening = False
            self.voice_btn.setText("◉")
            self.voice_btn.setStyleSheet("")
            return

        try:
            import speech_recognition as sr
        except ImportError:
            self._show_toast("Install speech_recognition: pip install SpeechRecognition", "warning")
            return

        self.voice_listening = True
        self.voice_btn.setText("●")
        self.voice_btn.setStyleSheet(
            "QPushButton { background-color: #ef4444; color: #ffffff; border: none; border-radius: 17px; }"
        )
        self._show_toast("Listening... speak now", "info")

        def listen_thread():
            recognizer = sr.Recognizer()
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)
                text = recognizer.recognize_google(audio)
                QTimer.singleShot(0, lambda t=text: self._insert_voice_text(t))
            except sr.WaitTimeoutError:
                QTimer.singleShot(0, lambda: self._show_toast("No speech detected", "warning"))
            except sr.UnknownValueError:
                QTimer.singleShot(0, lambda: self._show_toast("Could not understand audio", "warning"))
            except sr.RequestError as e:
                QTimer.singleShot(0, lambda: self._show_toast(f"Speech API error: {e}", "error"))
            except Exception as e:
                QTimer.singleShot(0, lambda: self._show_toast(f"Mic error: {e}", "error"))
            finally:
                self.voice_listening = False
                QTimer.singleShot(0, lambda: (
                    self.voice_btn.setText("◉"),
                    self.voice_btn.setStyleSheet(""),
                ))

        threading.Thread(target=listen_thread, daemon=True).start()

    def _insert_voice_text(self, text: str):
        current = self.input_box.text()
        if current:
            self.input_box.setText(current + " " + text)
        else:
            self.input_box.setText(text)
        self._show_toast("Speech captured", "info")

    def _speak_text(self, text: str):
        def tts_thread():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 170)
                clean = re.sub(r'[#*`_~\[\]()]', '', text)
                clean = re.sub(r'\n+', '. ', clean)
                engine.say(clean[:2000])
                engine.runAndWait()
            except ImportError:
                QTimer.singleShot(0, lambda: self._show_toast("Install pyttsx3: pip install pyttsx3", "warning"))
            except Exception as e:
                QTimer.singleShot(0, lambda: self._show_toast(f"TTS error: {e}", "error"))
        threading.Thread(target=tts_thread, daemon=True).start()

    # ── Message rendering ──────────────────────────────────────────────────

    def add_message(self, role, text):
        c = COLORS[self.current_theme]
        is_user = role == "user"
        is_loading = role == "loading"
        is_assistant = role == "assistant" or is_loading

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(4)

        # Bubble
        bubble = QFrame()
        if is_user:
            bubble.setObjectName("UserBubble")
            bubble.setMaximumWidth(480)
        else:
            bubble.setObjectName("AssistantBubble")

        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(14, 10, 14, 10)
        bubble_layout.setSpacing(0)

        textbox = QTextEdit()
        textbox.setReadOnly(True)
        textbox.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        textbox.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        textbox.setFrameShape(QFrame.NoFrame)
        textbox.setFont(QFont("Segoe UI", 12))
        textbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        if is_user:
            textbox.setStyleSheet(f"background: transparent; color: {c['user_text']}; border: none;")
        elif is_loading:
            textbox.setStyleSheet(f"background: transparent; color: {c['subtext']}; border: none;")
        else:
            textbox.setStyleSheet(f"background: transparent; color: {c['assistant_text']}; border: none;")

        self._render_markdown(textbox, text)
        bubble_layout.addWidget(textbox)

        if is_user:
            h_layout = QHBoxLayout()
            h_layout.setContentsMargins(0, 0, 0, 0)
            h_layout.addStretch()
            h_layout.addWidget(bubble)
            container_layout.addLayout(h_layout)
        else:
            container_layout.addWidget(bubble)

        # Deferred resize after layout settles
        QTimer.singleShot(20, lambda tb=textbox: self._resize_textbox(tb))
        QTimer.singleShot(150, lambda tb=textbox: self._resize_textbox(tb))

        # Action buttons for assistant — subtle and compact
        if role == "assistant":
            btn_row = QHBoxLayout()
            btn_row.setContentsMargins(4, 0, 0, 0)
            btn_row.setSpacing(2)

            def _regenerate():
                for msg in reversed(self.message_history):
                    if msg["role"] == "user":
                        if self.message_history and self.message_history[-1]["role"] == "assistant":
                            self.message_history.pop()
                        threading.Thread(target=self._generate, args=(msg["content"],), daemon=True).start()
                        return

            def _copy_text():
                QApplication.clipboard().setText(text)
                self._show_toast("Copied!", "info")

            def _branch_here():
                idx = len(self.message_history) - 1
                for i in range(len(self.message_history) - 1, -1, -1):
                    if self.message_history[i]["role"] == "assistant" and self.message_history[i]["content"] == text:
                        idx = i
                        break
                self._branch_conversation(idx)

            def _speak_response():
                self._speak_text(text)

            for label, callback in [
                ("Regenerate", _regenerate),
                ("Copy", _copy_text),
                ("Branch", _branch_here),
                ("Speak", _speak_response),
            ]:
                btn = QPushButton(label)
                btn.setObjectName("ActionBtn")
                btn.clicked.connect(callback)
                btn_row.addWidget(btn)

            btn_row.addStretch()
            container_layout.addLayout(btn_row)

        # Edit on double-click for user messages
        if is_user:
            def _edit_message(event):
                from PySide6.QtWidgets import QInputDialog
                current_text = textbox.toPlainText()
                new_text, ok = QInputDialog.getText(self.root, "Edit Message", "Edit your message:", text=current_text)
                if ok and new_text and new_text.strip():
                    for i in range(len(self.message_history) - 1, -1, -1):
                        if self.message_history[i]["role"] == "user" and self.message_history[i]["content"] == current_text:
                            self.message_history[i]["content"] = new_text.strip()
                            self.message_history = self.message_history[:i + 1]
                            break
                    self._render_markdown(textbox, new_text.strip())
                    QTimer.singleShot(10, lambda: self._resize_textbox(textbox))
                    threading.Thread(target=self._generate, args=(new_text.strip(),), daemon=True).start()

            textbox.mouseDoubleClickEvent = _edit_message

        # Insert before the stretch at the end
        count = self.chat_layout.count()
        self.chat_layout.insertWidget(count - 1, container)

        self._last_textbox = textbox
        self._last_container = container

        self._scroll_chat("bottom")
        return container

    # ── Clear helpers ──────────────────────────────────────────────────

    def _clear_chat_area(self):
        """Remove all messages and re-add the stretch."""
        self._safe_destroy_children(self.chat_frame)
        self.chat_layout.addStretch()

    # ── Markdown renderer ──────────────────────────────────────────────────

    def _render_markdown(self, textbox: QTextEdit, text: str):
        """Render markdown-formatted text into QTextEdit with formatting."""
        c = COLORS[self.current_theme]
        textbox.setReadOnly(False)
        textbox.clear()

        cursor = textbox.textCursor()
        cursor.movePosition(QTextCursor.Start)

        # Define formats
        base_text = c["user_text"] if textbox.styleSheet().find(c["user_text"]) != -1 else c["assistant_text"]

        fmt_normal = QTextCharFormat()
        fmt_normal.setForeground(QColor(base_text))
        fmt_normal.setFont(QFont("Segoe UI", 13))

        fmt_bold = QTextCharFormat()
        fmt_bold.setForeground(QColor(base_text))
        fmt_bold.setFont(QFont("Segoe UI", 13, QFont.Bold))

        fmt_italic = QTextCharFormat()
        fmt_italic.setForeground(QColor(base_text))
        font_it = QFont("Segoe UI", 13)
        font_it.setItalic(True)
        fmt_italic.setFont(font_it)

        fmt_code_inline = QTextCharFormat()
        fmt_code_inline.setForeground(QColor(c["green"]))
        fmt_code_inline.setBackground(QColor(c["code_bg"]))
        fmt_code_inline.setFont(QFont("Cascadia Code", 12))

        fmt_code_block = QTextCharFormat()
        fmt_code_block.setForeground(QColor(c["code_fg"]))
        fmt_code_block.setBackground(QColor(c["code_bg"]))
        fmt_code_block.setFont(QFont("Cascadia Code", 11))

        fmt_h1 = QTextCharFormat()
        fmt_h1.setForeground(QColor(c["accent"]))
        fmt_h1.setFont(QFont("Segoe UI", 17, QFont.Bold))

        fmt_h2 = QTextCharFormat()
        fmt_h2.setForeground(QColor(c["accent"]))
        fmt_h2.setFont(QFont("Segoe UI", 15, QFont.Bold))

        fmt_h3 = QTextCharFormat()
        fmt_h3.setForeground(QColor(c["accent"]))
        fmt_h3.setFont(QFont("Segoe UI", 13, QFont.Bold))

        # Split code blocks
        segments = re.split(r'(```(?:\w*)\n[\s\S]*?```)', text)

        first_segment = True
        for seg in segments:
            cb_match = re.match(r'```(\w*)\n([\s\S]*?)```', seg)
            if cb_match:
                code = cb_match.group(2)
                if not first_segment:
                    cursor.insertText("\n", fmt_normal)
                cursor.insertText(code, fmt_code_block)
                first_segment = False
            else:
                lines = seg.split("\n")
                for i, line in enumerate(lines):
                    if not first_segment or i > 0:
                        cursor.insertText("\n", fmt_normal)
                    first_segment = False
                    self._insert_markdown_line(cursor, line, fmt_normal, fmt_bold,
                                                fmt_italic, fmt_code_inline,
                                                fmt_h1, fmt_h2, fmt_h3)

        textbox.setReadOnly(True)

    def _insert_markdown_line(self, cursor, line, fmt_normal, fmt_bold,
                               fmt_italic, fmt_code_inline, fmt_h1, fmt_h2, fmt_h3):
        if line.startswith("### "):
            cursor.insertText(line[4:], fmt_h3)
            return
        if line.startswith("## "):
            cursor.insertText(line[3:], fmt_h2)
            return
        if line.startswith("# "):
            cursor.insertText(line[2:], fmt_h1)
            return
        if re.match(r'^[\-\*] ', line):
            cursor.insertText("  •  " + line[2:], fmt_normal)
            return

        # Inline formatting: **bold**, *italic*, `code`
        pattern = re.compile(r'(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)')
        pos = 0
        for m in pattern.finditer(line):
            if m.start() > pos:
                cursor.insertText(line[pos:m.start()], fmt_normal)
            if m.group(0).startswith("**"):
                cursor.insertText(m.group(2), fmt_bold)
            elif m.group(0).startswith("*"):
                cursor.insertText(m.group(3), fmt_italic)
            elif m.group(0).startswith("`"):
                cursor.insertText(m.group(4), fmt_code_inline)
            pos = m.end()
        cursor.insertText(line[pos:], fmt_normal)

    def _resize_textbox(self, textbox: QTextEdit):
        """Auto-size QTextEdit to fit content with deferred calculation."""
        try:
            doc = textbox.document()
            w = textbox.viewport().width()
            if w < 100:
                w = 500
            doc.setTextWidth(w)
            margins = textbox.contentsMargins()
            height = int(doc.size().height()) + margins.top() + margins.bottom() + 16
            textbox.setFixedHeight(max(36, min(height, 600)))
        except Exception:
            pass

    @staticmethod
    def _set_textbox_text(textbox: QTextEdit, text: str):
        """Update text in a QTextEdit during streaming."""
        try:
            textbox.setReadOnly(False)
            textbox.clear()
            cursor = textbox.textCursor()
            cursor.insertText(text)
            textbox.setReadOnly(True)
        except Exception:
            pass
