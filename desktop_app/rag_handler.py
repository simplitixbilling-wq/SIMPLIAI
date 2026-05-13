"""RAG handler mixin — RAG CRUD, sidebar, retrieval, web search, file upload (PySide6)."""

import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote, urlparse, parse_qs, unquote

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QSlider, QRadioButton, QButtonGroup,
    QFrame, QWidget, QInputDialog,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from utils import app_data_path
from system_tools import TokenOptimizer


class RAGHandlerMixin:
    """Mixin providing RAG database management and web/file helpers for SimpleAiagentAPP."""

    def _resolve_search_result_url(self, raw_url: str) -> str:
        if not raw_url:
            return ""
        if raw_url.startswith("//"):
            raw_url = "https:" + raw_url
        try:
            parsed = urlparse(raw_url)
            if "duckduckgo.com" in parsed.netloc:
                target = parse_qs(parsed.query).get("uddg", [""])[0]
                if target:
                    return unquote(target)
        except Exception:
            pass
        return raw_url

    def _fetch_web_excerpt(self, url: str, query: str, max_chars: int = 320) -> str:
        if not url or not url.startswith(("http://", "https://")):
            return ""
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0 Safari/537.36"
                )
            }
            response = requests.get(url, headers=headers, timeout=6)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                tag.decompose()

            query_terms = [term.lower() for term in re.findall(r"\b\w{3,}\b", query)]
            candidates = []
            for block in soup.find_all(["p", "li", "article", "main", "section"]):
                text = " ".join(block.get_text(" ", strip=True).split())
                if len(text) < 40:
                    continue
                score = sum(1 for term in query_terms if term in text.lower())
                if score or len(candidates) < 6:
                    candidates.append((score, text))

            if not candidates:
                page_text = " ".join(soup.get_text(" ", strip=True).split())
                return page_text[:max_chars]

            candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
            excerpt_parts = []
            total = 0
            for _score, text in candidates[:3]:
                remaining = max_chars - total
                if remaining <= 0:
                    break
                chunk = text[:remaining].strip()
                if chunk:
                    excerpt_parts.append(chunk)
                    total += len(chunk) + 1
            return " ".join(excerpt_parts)[:max_chars]
        except Exception:
            return ""

    def _init_rag_async(self):
        def _worker():
            try:
                from rag_manager import RAGManager
                mgr = RAGManager(base_directory=app_data_path("rag_databases"))
                self.rag_manager = mgr
                if not getattr(self, '_shutting_down', False):
                    self.signals.run_on_main.emit(self._refresh_rag_list)
            except Exception as e:
                if not getattr(self, '_shutting_down', False):
                    self.signals.update_status.emit(f"RAG init error: {e}")

        threading.Thread(target=_worker, daemon=True).start()

    # ── File upload ────────────────────────────────────────────────

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.root, "Upload File", "",
            "All Supported (*.txt *.pdf *.docx *.csv *.xlsx *.xls);;"
            "Text files (*.txt);;PDF files (*.pdf);;Word files (*.docx);;"
            "CSV files (*.csv);;Excel files (*.xlsx *.xls)"
        )
        if not file_path:
            return

        content = ""
        file_name = os.path.basename(file_path)

        try:
            self.update_status(f"📂 Loading {file_name}...")

            if file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

            elif file_path.endswith(".pdf"):
                try:
                    import fitz
                    doc = fitz.open(file_path)
                    max_pages = min(len(doc), 50)
                    for i in range(max_pages):
                        try:
                            page = doc[i]
                            text = page.get_text("text")
                            if text and text.strip():
                                content += text + "\n"
                            else:
                                try:
                                    ocr_text = page.get_textpage_ocr().extractText()
                                    if ocr_text and ocr_text.strip():
                                        content += ocr_text + "\n"
                                except Exception:
                                    pass
                        except Exception:
                            continue
                    doc.close()
                except Exception as e:
                    self.update_status(f"PDF Error: {str(e)[:100]}")
                    return

            elif file_path.endswith(".csv"):
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    content = df.to_string()
                except Exception as e:
                    self.update_status(f"CSV Error: {e}")
                    return

            elif file_path.endswith((".xlsx", ".xls")):
                try:
                    import pandas as pd
                    df = pd.read_excel(file_path)
                    content = df.to_string()
                except Exception as e:
                    self.update_status(f"Excel Error: {e}")
                    return

            elif file_path.endswith(".docx"):
                try:
                    from docx import Document
                    doc = Document(file_path)
                    for paragraph in doc.paragraphs:
                        if paragraph.text.strip():
                            content += paragraph.text + "\n"
                except Exception as e:
                    self.update_status(f"DOCX Error: {e}")
                    return

            if not content.strip():
                self.update_status("⚠️ File is empty or couldn't be read")
                self.add_message("assistant", "❌ File appears to be empty or unreadable.")
                return

            self.uploaded_content = content[:30000]
            file_size_kb = len(content) / 1024

            self.update_status(f"✓ Loaded: {file_name} ({file_size_kb:.1f} KB)")
            self.add_message(
                "assistant",
                f"📄 {file_name} uploaded successfully!\n\n"
                f"Size: {file_size_kb:.1f} KB\n\n"
                "Ask me anything about this file or use web search for additional context."
            )

            try:
                self._create_temporary_rag_for_uploaded_file(file_name)
            except Exception as rag_err:
                print(f"[TEMP RAG] {rag_err}")

        except Exception as e:
            self.update_status(f"❌ Error: {str(e)}")
            self.add_message("assistant", f"❌ Error loading file: {str(e)}")

    # ── Chunk retrieval ────────────────────────────────────────────

    def get_relevant_chunk(self, query):
        if not hasattr(self, "uploaded_content") or not self.uploaded_content:
            return ""

        model_ctx = getattr(self.model, "n_ctx", 2048) if self.model else 2048
        if callable(model_ctx):
            try:
                model_ctx = model_ctx()
            except Exception:
                model_ctx = 2048
        try:
            model_ctx = int(model_ctx) if model_ctx else 2048
        except Exception:
            model_ctx = 2048

        budget = TokenOptimizer.build_token_budget(model_ctx)
        max_context_chars = budget["max_context"] * 4

        rag_name = getattr(self, "temp_rag_db_name", None)

        if rag_name and self.rag_manager and rag_name in self.rag_manager.databases:
            try:
                results = self.rag_manager.retrieve(rag_name, query, k=3)
                if results and len(results) > 0:
                    best_chunk = results[0][0] if isinstance(results[0], tuple) else results[0]
                    if isinstance(best_chunk, str) and best_chunk.strip():
                        optimized = TokenOptimizer.optimize_for_small_context(
                            best_chunk, query, max_tokens=budget["max_context"])
                        return optimized if optimized else best_chunk.strip()[:max_context_chars]
            except Exception as e:
                print(f"[RAG ERROR] {e}")

        # SMART FALLBACK
        try:
            lines = self.uploaded_content.split("\n")
            query_lower = query.lower()
            query_words = [w.lower() for w in query.split() if len(w) > 3]

            generic_keywords = [
                "summary", "summarize", "key point", "overview", "main",
                "important", "highlight", "brief", "tell me about", "what is",
            ]
            is_generic = (any(kw in query_lower for kw in generic_keywords)
                          or len(query_words) < 2)

            if is_generic:
                return self.uploaded_content.strip()[:max_context_chars]

            matched = []
            total_chars = 0
            for line in lines:
                if line.strip() and any(word in line.lower() for word in query_words):
                    if total_chars + len(line) > max_context_chars:
                        break
                    matched.append(line.strip())
                    total_chars += len(line)

            if matched:
                return "\n".join(matched)

            return self.uploaded_content.strip()[:max_context_chars]
        except Exception as e:
            print(f"[CONTENT ERROR] {e}")
            return ""

    def _create_temporary_rag_for_uploaded_file(self, file_name):
        from rag_manager import RAGDatabase
        import uuid

        if not hasattr(self, "uploaded_content") or not self.uploaded_content:
            raise ValueError("No uploaded content to create RAG from")

        old_temp_name = getattr(self, "temp_rag_db_name", None)
        if old_temp_name and self.rag_manager and old_temp_name in self.rag_manager.databases:
            del self.rag_manager.databases[old_temp_name]

        temp_name = f"uploaded_temp_{uuid.uuid4().hex[:8]}"
        self.temp_rag_db_name = temp_name

        db = RAGDatabase(name=temp_name)
        if hasattr(self.rag_manager, "_chunk_text"):
            chunks = self.rag_manager._chunk_text(
                self.uploaded_content, chunk_size=400, chunk_overlap=80)
        else:
            text = self.uploaded_content
            chunks = [text[i : i + 400] for i in range(0, len(text), 320)]

        db.add_chunks(chunks)
        self.rag_manager.databases[temp_name] = db

    # ── Web search ─────────────────────────────────────────────────

    def toggle_web_search(self):
        self.web_search_enabled = not self.web_search_enabled
        if self.web_search_enabled:
            active_bg = "#6366f1" if self.current_theme == "Dark" else "#6366f1"
            active_border = "#818cf8" if self.current_theme == "Dark" else "#4f46e5"
            self.web_search_button.setStyleSheet(
                f"QPushButton {{ background-color: {active_bg}; color: #ffffff; "
                f"border: 1px solid {active_border}; border-radius: 17px; }}")
            self.update_status("🌐 Web Search: ON")
            self.send_btn.setEnabled(True)
        else:
            self.web_search_button.setStyleSheet("")
            self.update_status("🌐 Web Search: OFF")
            self.send_btn.setEnabled(
                self.model is not None and callable(self.model))

    def search_web(self, query, num_results=3):
        try:
            self.update_status("🔍 Searching web...")
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/91.0.4472.124 Safari/537.36"
                )
            }
            url = f"https://duckduckgo.com/html/?q={quote(query)}"
            response = requests.get(url, headers=headers, timeout=8)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            results = []
            seen = set()
            for result in soup.find_all("div", class_="result"):
                try:
                    title_elem = result.find("a", class_="result__a")
                    snippet_elem = result.find(class_="result__snippet")
                    if title_elem and snippet_elem:
                        title = title_elem.get_text().strip()
                        link = self._resolve_search_result_url(
                            title_elem.get("href", "").strip())
                        snippet = snippet_elem.get_text().strip()
                        snippet = snippet.replace("\n", " ")[:200]
                        if title and snippet:
                            key = (title.lower(), link)
                            if key in seen:
                                continue
                            seen.add(key)
                            entry = f"• **{title}**"
                            if link:
                                entry += f" ({link})"
                            entry += f": {snippet}"
                            excerpt = self._fetch_web_excerpt(link, query)
                            if excerpt:
                                entry += f"\n  Excerpt: {excerpt}"
                            results.append(entry)
                        if len(results) >= max(3, num_results):
                            break
                except Exception:
                    continue

            if results:
                return "\n\n".join(results)
            return ("🔎 No web results found. Try rephrasing your query "
                    "or check your internet connection.")

        except requests.exceptions.ConnectTimeout:
            return "⚠️ Connection timeout. Check your internet connection."
        except requests.exceptions.ConnectionError:
            return "⚠️ Connection error. Please check your internet connection."
        except requests.exceptions.Timeout:
            return "⏱️ Web search timed out. Please try again."
        except Exception as e:
            return f"⚠️ Web search unavailable: {str(e)[:100]}"

    # ── RAG CRUD ───────────────────────────────────────────────────

    def _open_rag_folder_dialog(self):
        self._open_rag_create_dialog()

    def _open_rag_create_dialog(self):
        dlg = QDialog(self.root)
        dlg.setWindowTitle("New RAG Database")
        dlg.setFixedSize(420, 440)
        layout = QVBoxLayout(dlg)

        # Name
        layout.addWidget(QLabel("Database name:"))
        name_entry = QLineEdit()
        name_entry.setPlaceholderText("e.g. my_docs")
        layout.addWidget(name_entry)

        # Source type
        layout.addWidget(QLabel("Source type:"))
        src_grp = QButtonGroup(dlg)
        src_row = QHBoxLayout()
        rb_folder = QRadioButton("📁 Folder")
        rb_folder.setChecked(True)
        rb_url = QRadioButton("🌐 Web URL")
        src_grp.addButton(rb_folder)
        src_grp.addButton(rb_url)
        src_row.addWidget(rb_folder)
        src_row.addWidget(rb_url)
        layout.addLayout(src_row)

        # Source path / URL
        layout.addWidget(QLabel("Folder path or URL:"))
        source_row = QHBoxLayout()
        source_entry = QLineEdit()
        source_entry.setPlaceholderText("Paste URL or click Browse")
        source_row.addWidget(source_entry, 1)

        def browse():
            path = QFileDialog.getExistingDirectory(
                dlg, "Select folder for RAG database")
            if path:
                source_entry.setText(path)
                rb_folder.setChecked(True)

        browse_btn = QPushButton("Browse")
        browse_btn.setObjectName("AccentBtn")
        browse_btn.setFixedWidth(72)
        browse_btn.clicked.connect(browse)
        source_row.addWidget(browse_btn)
        layout.addLayout(source_row)

        # Chunk size slider
        chunk_lbl = QLabel("Chunk size: 512 chars")
        layout.addWidget(chunk_lbl)
        chunk_slider = QSlider(Qt.Horizontal)
        chunk_slider.setRange(128, 1024)
        chunk_slider.setValue(512)
        chunk_slider.valueChanged.connect(
            lambda v: chunk_lbl.setText(f"Chunk size: {v} chars"))
        layout.addWidget(chunk_slider)

        # Overlap slider
        overlap_lbl = QLabel("Overlap: 100 chars")
        layout.addWidget(overlap_lbl)
        overlap_slider = QSlider(Qt.Horizontal)
        overlap_slider.setRange(0, 256)
        overlap_slider.setValue(100)
        overlap_slider.valueChanged.connect(
            lambda v: overlap_lbl.setText(f"Overlap: {v} chars"))
        layout.addWidget(overlap_slider)

        status_lbl = QLabel("")
        status_lbl.setStyleSheet("color: #F38BA8;")
        layout.addWidget(status_lbl)

        layout.addStretch()

        def create():
            rag_name = name_entry.text().strip()
            source = source_entry.text().strip()
            stype = "folder" if rb_folder.isChecked() else "url"
            csize = chunk_slider.value()
            coverlap = overlap_slider.value()
            if not rag_name:
                status_lbl.setText("❌ Enter a database name")
                return
            if not source:
                status_lbl.setText("❌ Enter a folder path or URL")
                return
            if self.rag_manager and rag_name in self.rag_manager.list_databases():
                status_lbl.setText(f"❌ '{rag_name}' already exists")
                return
            dlg.accept()
            if stype == "url":
                self.update_status("⏳ Importing from URL...")
                threading.Thread(
                    target=self._create_rag_url_thread,
                    args=(source, rag_name, csize, coverlap),
                    daemon=True,
                ).start()
            else:
                if not os.path.isdir(source):
                    self.update_status("❌ Folder not found")
                    return
                self.update_status("⏳ Creating RAG database...")
                threading.Thread(
                    target=self._create_rag_database_thread,
                    args=(source, rag_name, csize, coverlap),
                    daemon=True,
                ).start()

        create_btn = QPushButton("✔ Create Database")
        create_btn.setObjectName("GreenBtn")
        create_btn.setFixedHeight(38)
        create_btn.setFont(QFont("Segoe UI", 13, QFont.Bold))
        create_btn.clicked.connect(create)
        layout.addWidget(create_btn)

        dlg.exec()

    def _create_rag_url_thread(self, url, rag_name, chunk_size=512, chunk_overlap=100):
        try:
            self.rag_manager.create_from_url(url, rag_name, chunk_size, chunk_overlap)
            info = self.rag_manager.get_database_info(rag_name)
            self.signals.update_status.emit(
                f"Imported '{rag_name}' from URL ({info['num_chunks']} chunks)")
            self.signals.run_on_main.emit(self._refresh_rag_list)
        except Exception as e:
            self.signals.update_status.emit(f"URL import error: {e}")

    def _reindex_rag_database(self, db_name):
        if self.rag_manager is None:
            self.update_status("RAG still loading, please wait")
            return
        self.update_status(f"🔄 Re-indexing '{db_name}'...")
        threading.Thread(
            target=self._reindex_rag_thread, args=(db_name,), daemon=True).start()

    def _reindex_rag_thread(self, db_name):
        try:
            self.rag_manager.reindex_database(db_name)
            info = self.rag_manager.get_database_info(db_name)
            self.signals.update_status.emit(
                f"Re-indexed '{db_name}' ({info['num_chunks']} chunks)")
            self.signals.run_on_main.emit(self._refresh_rag_list)
        except Exception as e:
            self.signals.update_status.emit(f"Re-index failed: {e}")

    def _create_rag_database_thread(self, folder_path, rag_name,
                                    chunk_size=512, chunk_overlap=100):
        try:
            self.signals.update_status.emit(f"Processing files from {folder_path}...")
            self.rag_manager.create_from_folder(
                folder_path, rag_name, chunk_size, chunk_overlap)
            info = self.rag_manager.get_database_info(rag_name)
            self.signals.update_status.emit(
                f"RAG '{rag_name}' created with {info['num_chunks']} chunks")
            self.signals.run_on_main.emit(self._refresh_rag_list)
        except Exception as e:
            self.signals.update_status.emit(f"Error creating RAG: {str(e)}")

    # ── RAG sidebar ────────────────────────────────────────────────

    def _refresh_rag_list(self):
        # Clear existing
        while self.rag_list_layout.count() > 1:
            item = self.rag_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.rag_widgets = {}

        if self.rag_manager is None:
            return

        databases = self.rag_manager.list_databases()
        if not databases:
            empty_lbl = QLabel("No RAG databases")
            empty_lbl.setObjectName("Muted")
            count = self.rag_list_layout.count()
            self.rag_list_layout.insertWidget(count - 1, empty_lbl)
            return

        for db_name in databases:
            info = self.rag_manager.get_database_info(db_name)
            self._add_rag_to_sidebar(db_name, info)

    def _add_rag_to_sidebar(self, db_name: str, info: Dict):
        row = QFrame()
        row.setObjectName("RagRow")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(8, 4, 5, 4)
        row_layout.setSpacing(3)

        label = QLabel(f"{db_name} ({info['num_chunks']})")
        label.setObjectName("StatusLabel")
        label.setFont(QFont("Segoe UI", 11))
        label.setWordWrap(True)
        row_layout.addWidget(label, 1)

        select_btn = QPushButton("✓")
        select_btn.setFixedSize(24, 24)
        if db_name == self.current_rag_database:
            select_btn.setObjectName("AccentBtn")
        else:
            select_btn.setObjectName("IconBtn")
        select_btn.clicked.connect(
            lambda checked, n=db_name: self._select_rag_database(n))
        row_layout.addWidget(select_btn)

        delete_btn = QPushButton("✕")
        delete_btn.setObjectName("IconBtn")
        delete_btn.setFixedSize(24, 24)
        delete_btn.clicked.connect(
            lambda checked, n=db_name: self._delete_rag_database(n))
        row_layout.addWidget(delete_btn)

        self.rag_widgets[db_name] = (row, label, select_btn, delete_btn)

        count = self.rag_list_layout.count()
        self.rag_list_layout.insertWidget(count - 1, row)

    def _select_rag_database(self, db_name):
        if self.current_rag_database == db_name:
            self.current_rag_database = None
            self.update_status("RAG deselected")
        else:
            self.current_rag_database = db_name
            self.update_status(f"RAG: {db_name}")

        if self.current_chat_id:
            if self.current_rag_database:
                self.chat_rag_settings[self.current_chat_id] = self.current_rag_database
            else:
                self.chat_rag_settings.pop(self.current_chat_id, None)
            self._save_chat_rag_settings()
        self._refresh_rag_list()

    def _delete_rag_database(self, db_name):
        confirm, ok = QInputDialog.getText(
            self.root,
            "Delete RAG Database",
            f"Type '{db_name}' to confirm deletion:",
        )
        if not ok or confirm != db_name:
            self.update_status("❌ Deletion cancelled")
            return
        try:
            if self.rag_manager is None:
                self.update_status("RAG still loading, please wait")
                return
            self.rag_manager.delete_database(db_name)
            if self.current_rag_database == db_name:
                self.current_rag_database = None
            self.update_status(f"✅ RAG database '{db_name}' deleted")
            self._refresh_rag_list()
        except Exception as e:
            self.update_status(f"❌ Error deleting RAG: {str(e)}")

    # ── RAG reference extraction ───────────────────────────────────

    def _extract_rag_references(self, text: str) -> Tuple[str, List[str]]:
        pattern = r"@(\w+)"
        matches = re.findall(pattern, text)
        rag_names = list(set(matches))
        clean_text = re.sub(pattern, "", text).strip()
        return clean_text, rag_names

    def _retrieve_rag_context(self, rag_names: List[str], query: str,
                              k: int = 5) -> Tuple[str, List[str]]:
        all_context = ""
        sources_used = []
        self.last_rag_hits = 0

        for rag_name in rag_names:
            if rag_name not in self.rag_manager.list_databases():
                all_context += f"\n⚠️ RAG database '{rag_name}' not found\n"
                continue

            # Use full knowledge.md as context for complete coverage
            knowledge_md = self.rag_manager.read_knowledge_markdown(rag_name)
            if knowledge_md:
                all_context += f"\n=== FULL KNOWLEDGE BASE [{rag_name}] ===\n"
                all_context += knowledge_md
                all_context += f"\n=== END KNOWLEDGE BASE [{rag_name}] ===\n"
                # Collect source files from database metadata
                db = self.rag_manager.databases.get(rag_name)
                if db:
                    seen = set()
                    for i in range(len(db.chunks)):
                        try:
                            src = db.get_chunk_source(i)
                            if src and src not in seen:
                                seen.add(src)
                                sources_used.append(src)
                        except (ValueError, IndexError):
                            pass
                if not sources_used:
                    sources_used.append(rag_name)
            else:
                all_context += f"\n⚠️ No relevant content found in [{rag_name}]\n"

        return all_context, sources_used
