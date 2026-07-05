"""Chat management mixin — CRUD, sidebar, system prompts, auto-save, export (PySide6)."""

import os
import threading

from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QMenu, QFileDialog, QTextEdit, QDialog,
    QWidget,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QCursor

from app_core.utils import app_data_path


class ChatManagerMixin:
    """Mixin providing chat management methods for SimpleAiagentAPP."""

    # ── Sidebar rows ───────────────────────────────────────────────

    def _add_chat_to_sidebar(self, chat_id):
        if not hasattr(self, "chat_widgets"):
            self.chat_widgets = {}

        row = QFrame()
        row.setObjectName("ChatRow")
        row.setCursor(QCursor(Qt.PointingHandCursor))
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(8, 6, 5, 6)
        row_layout.setSpacing(4)

        label = QLabel(f"💬 {chat_id}")
        label.setWordWrap(True)
        label.setFont(QFont("Segoe UI", 12))
        label.mousePressEvent = lambda e, cid=chat_id: self._load_chat(cid)
        row_layout.addWidget(label, 1)

        menu_btn = QPushButton("⋮")
        menu_btn.setObjectName("IconBtn")
        menu_btn.setFixedSize(25, 25)
        menu_btn.setFont(QFont("Segoe UI", 14))
        menu_btn.clicked.connect(
            lambda checked, cid=chat_id, btn=menu_btn: self._open_chat_menu(cid, btn))
        row_layout.addWidget(menu_btn)

        self.chat_widgets[chat_id] = (row, label)

        # Insert before final stretch
        count = self.chat_list_layout.count()
        self.chat_list_layout.insertWidget(count - 1, row)

        self._filter_chat_list()

    def _inline_rename_chat(self, chat_id):
        if chat_id not in self.chat_widgets:
            return
        row, old_label = self.chat_widgets[chat_id]
        old_label.hide()

        entry = QLineEdit()
        entry.setText(chat_id)
        row.layout().insertWidget(0, entry)
        entry.setFocus()
        entry.selectAll()

        def save():
            new_name = entry.text().strip()
            if not new_name or new_name == chat_id:
                cancel()
                return
            if new_name in self.chats:
                self.update_status("Name already exists")
                cancel()
                return

            self.chats[new_name] = self.chats.pop(chat_id)
            if self.current_chat_id == chat_id:
                self.current_chat_id = new_name
            self.chat_db.rename_chat(chat_id, new_name)

            if chat_id in self.chat_rag_settings:
                self.chat_rag_settings[new_name] = self.chat_rag_settings.pop(chat_id)
            if chat_id in self.chat_system_prompts:
                self.chat_system_prompts[new_name] = self.chat_system_prompts.pop(chat_id)

            entry.deleteLater()
            self._clear_chat_list()
            self._load_and_sort_chats()
            self.update_status(f"Renamed to {new_name}")

        def cancel():
            entry.deleteLater()
            old_label.show()

        entry.returnPressed.connect(save)
        # Escape via key event override
        _orig_key = entry.keyPressEvent

        def _key(event):
            if event.key() == Qt.Key_Escape:
                cancel()
            else:
                _orig_key(event)

        entry.keyPressEvent = _key

    def _load_chat(self, chat_id):
        self.current_chat_id = chat_id
        self.message_history = self.chats[chat_id]

        self._clear_chat_area()

        for msg in self.message_history:
            self.add_message(msg["role"], msg["content"])

        self.current_rag_database = self.chat_rag_settings.get(chat_id)
        self._refresh_rag_list()
        self._scroll_chat("bottom")

    def _load_saved_chats(self):
        os.makedirs(app_data_path("saved_chats"), exist_ok=True)
        self.chat_widgets = {}
        self._load_chat_rag_settings()
        self._load_chat_system_prompts()
        self._load_and_sort_chats()

        max_chat_number = 0
        for chat_id in self.chats.keys():
            if chat_id.startswith("Chat "):
                try:
                    num = int(chat_id.replace("Chat ", ""))
                    max_chat_number = max(max_chat_number, num)
                except ValueError:
                    continue

        self.chat_counter = max(max_chat_number + 1, 1)

        if getattr(self, "sorted_chat_ids", None):
            first_chat = self.sorted_chat_ids[0]
            self.current_chat_id = first_chat
            self.message_history = self.chats.get(first_chat, [])
            self._load_chat(first_chat)
        elif not self.chats:
            self.new_chat()

    # ── Per-chat RAG settings ──────────────────────────────────────

    def _load_chat_rag_settings(self):
        self.chat_rag_settings = self.chat_db.get_all_meta("rag_db")

    def _save_chat_rag_settings(self):
        existing = self.chat_db.get_all_meta("rag_db")
        for cid in existing:
            if cid not in self.chat_rag_settings:
                self.chat_db.delete_meta(cid, "rag_db")
        for cid, val in self.chat_rag_settings.items():
            self.chat_db.set_meta(cid, "rag_db", val)

    # ── Chat list helpers ──────────────────────────────────────────

    def _clear_chat_list(self):
        """Remove all chat row widgets from the sidebar list."""
        while self.chat_list_layout.count() > 1:  # keep stretch
            item = self.chat_list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.chat_widgets = {}

    def _load_and_sort_chats(self):
        self.chats = self.chat_db.load_all_chats()
        self.sorted_chat_ids = self.chat_db.sorted_chat_ids()

        self._clear_chat_list()

        for chat_id in self.sorted_chat_ids:
            self._add_chat_to_sidebar(chat_id)

        self._filter_chat_list()

    # ── Context menu ───────────────────────────────────────────────

    def _open_chat_menu(self, chat_id, button):
        menu = QMenu(self.root)

        menu.addAction("💾 Save", lambda: self._save_chat(chat_id))
        menu.addAction("📄 Export .txt", lambda: self._export_chat(chat_id))
        menu.addAction("✏ Rename", lambda: self._inline_rename_chat(chat_id))
        menu.addAction("🗑 Delete", lambda: self._delete_chat(chat_id))

        menu.exec(button.mapToGlobal(button.rect().bottomLeft()))

    def _save_chat(self, chat_id):
        self.chat_db.save_chat(chat_id, self.chats[chat_id])
        self._clear_chat_list()
        self._load_and_sort_chats()
        self.update_status(f"Saved {chat_id}")

    def _delete_chat(self, chat_id):
        if chat_id in self.chats:
            del self.chats[chat_id]
        if chat_id in self.chat_rag_settings:
            del self.chat_rag_settings[chat_id]
        if chat_id in self.chat_system_prompts:
            del self.chat_system_prompts[chat_id]

        self.chat_db.delete_chat(chat_id)
        self._clear_chat_list()
        self._load_and_sort_chats()
        self.update_status(f"Deleted {chat_id}")

    # ── Chat search / filter ───────────────────────────────────────

    def _filter_chat_list(self, text=None):
        if not hasattr(self, "chat_widgets") or not self.chat_widgets:
            return
        raw = (text if text is not None else self.chat_search_box.text()).strip().lower()
        query_norm = raw.replace(" ", "")
        for chat_id, (row, _) in list(self.chat_widgets.items()):
            try:
                name_norm = chat_id.lower().replace(" ", "")
                if raw == "" or raw in chat_id.lower() or query_norm in name_norm:
                    row.show()
                else:
                    row.hide()
            except Exception:
                continue

    # ── Export ──────────────────────────────────────────────────────

    def _export_chat(self, chat_id=None):
        if chat_id is None:
            chat_id = self.current_chat_id
        messages = self.chats.get(chat_id, [])
        if not messages:
            self.update_status("Nothing to export")
            return

        path, _ = QFileDialog.getSaveFileName(
            self.root, "Export chat", f"{chat_id}.txt",
            "Text files (*.txt);;All files (*.*)"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Chat: {chat_id}\n{'=' * 60}\n\n")
            for msg in messages:
                role = "You" if msg["role"] == "user" else "AI"
                f.write(f"[{role}]\n{msg['content']}\n\n")
        self.update_status(f"Exported to {os.path.basename(path)}")

    # ── Token counter ──────────────────────────────────────────────

    def _update_token_counter(self, text=None):
        if text is None:
            text = self.input_box.text()
        tokens = max(1, len(text) // 4) if text else 0
        self.token_counter_label.setText(f"{tokens}t" if tokens else "")

    # ── Chat IDs / branching ───────────────────────────────────────

    def _get_next_chat_id(self):
        max_chat_number = 0
        for chat_id in self.chats.keys():
            if chat_id.startswith("Chat "):
                try:
                    num = int(chat_id.replace("Chat ", ""))
                    max_chat_number = max(max_chat_number, num)
                except ValueError:
                    continue
        return f"Chat {max_chat_number + 1}"

    def _branch_conversation(self, branch_at_index):
        if not self.current_chat_id or not self.message_history:
            return
        branched = [dict(m) for m in self.message_history[: branch_at_index + 1]]
        new_id = self._get_next_chat_id()
        self.chats[new_id] = branched
        self.chat_db.save_chat(new_id, branched)

        if self.current_chat_id in self.chat_rag_settings:
            self.chat_rag_settings[new_id] = self.chat_rag_settings[self.current_chat_id]
            self.chat_db.set_meta(new_id, "rag_db", self.chat_rag_settings[new_id])
        if self.current_chat_id in self.chat_system_prompts:
            self.chat_system_prompts[new_id] = self.chat_system_prompts[self.current_chat_id]
            self.chat_db.set_meta(new_id, "system_prompt", self.chat_system_prompts[new_id])

        self._add_chat_to_sidebar(new_id)
        self._load_chat(new_id)
        self._show_toast(f"Branched to {new_id}", "info")

    def new_chat(self):
        chat_id = self._get_next_chat_id()
        self.current_chat_id = chat_id
        self.message_history = []
        self.chats[chat_id] = self.message_history

        self._clear_chat_area()
        self.add_message("assistant", "🤖 New chat started. How can I help you today?")
        self._scroll_chat("top")

        try:
            self._add_chat_to_sidebar(chat_id)
        except Exception as e:
            self.update_status(f"Error while updating chat list: {e}")

        self.chat_counter = int(chat_id.replace("Chat ", "")) + 1

    # ── Per-chat system prompt ─────────────────────────────────────

    def _load_chat_system_prompts(self):
        self.chat_system_prompts = self.chat_db.get_all_meta("system_prompt")

    def _save_chat_system_prompts(self):
        existing = self.chat_db.get_all_meta("system_prompt")
        for cid in existing:
            if cid not in self.chat_system_prompts:
                self.chat_db.delete_meta(cid, "system_prompt")
        for cid, val in self.chat_system_prompts.items():
            self.chat_db.set_meta(cid, "system_prompt", val)

    def _open_system_prompt_dialog(self):
        if not self.current_chat_id:
            self._show_toast("Open or create a chat first.", "warn")
            return
        cid = self.current_chat_id
        existing = self.chat_system_prompts.get(cid, "")

        dlg = QDialog(self.root)
        dlg.setWindowTitle(f"✏ System Prompt — {cid}")
        dlg.resize(480, 320)
        layout = QVBoxLayout(dlg)

        hint = QLabel(
            "Custom system prompt for this chat.\n"
            "Leave blank to use the default behaviour."
        )
        hint.setObjectName("Muted")
        layout.addWidget(hint)

        txt = QTextEdit()
        txt.setFont(QFont("Segoe UI", 12))
        txt.setPlainText(existing)
        layout.addWidget(txt, 1)

        btn_row = QHBoxLayout()

        def save():
            content = txt.toPlainText().strip()
            if content:
                self.chat_system_prompts[cid] = content
            else:
                self.chat_system_prompts.pop(cid, None)
            self._save_chat_system_prompts()
            dlg.accept()
            status = "set" if content else "cleared"
            self._show_toast(f"System prompt {status} for {cid}.", "info")

        def clear():
            txt.clear()

        save_btn = QPushButton("💾 Save")
        save_btn.setObjectName("GreenBtn")
        save_btn.clicked.connect(save)
        btn_row.addWidget(save_btn, 1)

        clear_btn = QPushButton("🗑 Clear")
        clear_btn.setObjectName("RedBtn")
        clear_btn.clicked.connect(clear)
        btn_row.addWidget(clear_btn, 1)
        layout.addLayout(btn_row)

        dlg.exec()

    # ── Auto-save ──────────────────────────────────────────────────

    def _auto_save_chat(self):
        if not self.current_chat_id:
            return
        self.chat_db.save_chat(self.current_chat_id, self.message_history)

    def _auto_save_timer_tick(self):
        """Periodic save — called by QTimer in agent.py."""
        try:
            if self.chats:
                self.chat_db.save_all_chats(self.chats)
        except Exception as e:
            print(f"[AUTO-SAVE] {e}")
