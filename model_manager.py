"""Model management mixin — scanning, loading, HF downloader, GPU detection (PySide6)."""

import gc
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import requests
from urllib.parse import quote

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QFrame, QScrollArea, QWidget,
    QProgressBar, QLineEdit,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from utils import app_data_path

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


class ModelManagerMixin:
    """Mixin providing model management methods for SimpleAiagentAPP."""

    def _get_torch_module(self):
        """Import torch lazily and disable it after the first failure."""
        if getattr(self, "_torch_unavailable", False):
            return None
        cached = getattr(self, "_torch_module", None)
        if cached is not None:
            return cached
        try:
            import torch
            self._torch_module = torch
            return torch
        except Exception:
            self._torch_unavailable = True
            sys.modules.pop("torch", None)
            return None

    def _model_display_label(self, filepath) -> str:
        p = Path(filepath)
        name = p.stem
        m = re.search(r"(Q\d[_A-Z0-9]*)", name, re.IGNORECASE)
        quant = m.group(1).upper() if m else ""
        try:
            size_gb = round(p.stat().st_size / (1024 ** 3), 1)
        except Exception:
            size_gb = 0.0
        if quant:
            return f"{name} ({quant} · {size_gb}GB)"
        return f"{name} ({size_gb}GB)"

    def _get_max_model_size_gb(self) -> float:
        available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        gpu_type = self.gpu_info.get("type", "CPU")
        vram_gb = self.gpu_info.get("vram", 0.0)

        if gpu_type == "APPLE_METAL":
            # Unified memory — model can use most of system RAM
            usable = available_ram_gb * 0.75
            return round(usable, 1)
        elif gpu_type in ("NVIDIA", "AMD") and vram_gb >= 2:
            effective_gb = vram_gb + min(available_ram_gb * 0.25, 3.0)
            return round(effective_gb * 0.90, 1)
        else:
            usable = min(available_ram_gb, float(self.system_ram)) * 0.65
            return round(usable, 1)

    # ── Per-model settings ─────────────────────────────────────────

    def _load_model_configs(self):
        self.model_configs = self.chat_db.get_kv("model_configs", {})

    def _save_model_configs(self):
        self.chat_db.set_kv("model_configs", self.model_configs)

    def _open_per_model_settings(self):
        if not self.model_path:
            return
        filename = Path(self.model_path).name
        saved = self.model_configs.get(filename, {})

        dlg = QDialog(self.root)
        dlg.setWindowTitle(f"⚙ Settings — {filename}")
        dlg.setFixedSize(400, 340)
        layout = QVBoxLayout(dlg)

        lbl = QLabel(f"Model: {filename}")
        lbl.setFont(QFont("Segoe UI", 12, QFont.Bold))
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        # Temperature
        layout.addWidget(QLabel("Temperature (creativity):"))
        temp_row = QHBoxLayout()
        temp_slider = QSlider(Qt.Horizontal)
        temp_slider.setRange(5, 150)
        temp_slider.setValue(int(saved.get("temperature", 0.25) * 100))
        temp_lbl = QLabel(f"{temp_slider.value() / 100:.2f}")
        temp_lbl.setFixedWidth(40)
        temp_slider.valueChanged.connect(
            lambda v: temp_lbl.setText(f"{v / 100:.2f}"))
        temp_row.addWidget(temp_slider, 1)
        temp_row.addWidget(temp_lbl)
        layout.addLayout(temp_row)

        # Context window
        layout.addWidget(QLabel("Context window (n_ctx):"))
        ctx_combo = QComboBox()
        ctx_choices = ["default", "1024", "2048", "4096", "8192"]
        ctx_combo.addItems(ctx_choices)
        saved_ctx = str(saved.get("n_ctx", ""))
        if saved_ctx in ctx_choices:
            ctx_combo.setCurrentText(saved_ctx)
        else:
            ctx_combo.setCurrentText("default")
        layout.addWidget(ctx_combo)

        # CPU threads
        cpu_max = os.cpu_count() or 8
        layout.addWidget(QLabel(f"CPU threads (1–{cpu_max}):"))
        thr_row = QHBoxLayout()
        thr_slider = QSlider(Qt.Horizontal)
        thr_slider.setRange(1, cpu_max)
        thr_slider.setValue(saved.get("n_threads", self.config["n_threads"]))
        thr_lbl = QLabel(str(thr_slider.value()))
        thr_lbl.setFixedWidth(30)
        thr_slider.valueChanged.connect(lambda v: thr_lbl.setText(str(v)))
        thr_row.addWidget(thr_slider, 1)
        thr_row.addWidget(thr_lbl)
        layout.addLayout(thr_row)

        layout.addStretch()

        # Buttons
        btn_row = QHBoxLayout()

        def save_settings():
            cfg = {
                "temperature": round(temp_slider.value() / 100, 3),
                "n_threads": thr_slider.value(),
            }
            if ctx_combo.currentText() != "default":
                cfg["n_ctx"] = int(ctx_combo.currentText())
            self.model_configs[filename] = cfg
            self._save_model_configs()
            dlg.accept()
            if self.model is not None and hasattr(self, "model_config"):
                self.model_config["temperature"] = cfg["temperature"]

        def reset_defaults():
            self.model_configs.pop(filename, None)
            self._save_model_configs()
            dlg.accept()

        save_btn = QPushButton("💾 Save")
        save_btn.setObjectName("GreenBtn")
        save_btn.clicked.connect(save_settings)
        btn_row.addWidget(save_btn, 1)

        reset_btn = QPushButton("↺ Reset")
        reset_btn.clicked.connect(reset_defaults)
        btn_row.addWidget(reset_btn, 1)
        layout.addLayout(btn_row)

        dlg.exec()

    # ── Model scanning / auto-select ───────────────────────────────

    def _scan_models_on_startup(self):
        models_dir = Path(app_data_path("models"))
        models_dir.mkdir(exist_ok=True)
        files = list(models_dir.glob("*.gguf"))

        if not files:
            self.model_menu.blockSignals(True)
            self.model_menu.clear()
            self.model_menu.addItem("No model found")
            self.model_menu.blockSignals(False)
            QTimer.singleShot(500, self._show_first_run_wizard)
            return

        self.model_map = {self._model_display_label(f): str(f) for f in files}
        names = list(self.model_map.keys())

        self.model_menu.blockSignals(True)
        self.model_menu.clear()
        self.model_menu.addItems(names)
        self.model_menu.setCurrentIndex(0)
        self.model_menu.blockSignals(False)
        self.model_path = self.model_map[names[0]]

    def _auto_select_model(self) -> str:
        if not hasattr(self, "model_map") or not self.model_map:
            return None
        names = list(self.model_map.keys())
        if len(names) == 1:
            return names[0]

        def model_score(name):
            n = name.lower()
            score = 0
            if "q8" in n: score += 80
            elif "q6" in n: score += 70
            elif "q5" in n: score += 60
            elif "q4" in n: score += 50
            elif "q3" in n: score += 30
            elif "q2" in n: score += 10
            if "-it" in n or "instruct" in n or "chat" in n:
                score += 20
            return score

        scored = [(name, model_score(name)) for name in names]
        scored.sort(key=lambda x: x[1], reverse=True)
        if self.system_ram <= 8:
            scored.sort(key=lambda x: x[1])
        return scored[0][0]

    # ── Model loading ──────────────────────────────────────────────

    def load_model(self):
        if not self.model_path:
            self._show_toast("No model selected.", "warn")
            return
        try:
            model_size_gb = round(Path(self.model_path).stat().st_size / (1024 ** 3), 2)
            max_safe_gb = self._get_max_model_size_gb()
            if model_size_gb > max_safe_gb:
                avail_gb = round(psutil.virtual_memory().available / (1024 ** 3), 1)
                self._confirm_oversize_load(model_size_gb, avail_gb, max_safe_gb)
                return
        except Exception:
            pass
        threading.Thread(target=self._load_model_thread, daemon=True).start()

    def _confirm_oversize_load(self, model_gb, avail_gb, max_gb):
        dlg = QDialog(self.root)
        dlg.setWindowTitle("⚠ RAM Warning")
        dlg.setFixedSize(400, 200)
        layout = QVBoxLayout(dlg)

        warn_lbl = QLabel(
            f"⚠ This model is {model_gb} GB but your system only has\n"
            f"{avail_gb} GB available (safe limit: {max_gb} GB).\n\n"
            "Loading may cause the app to freeze or crash.\nLoad anyway?"
        )
        warn_lbl.setWordWrap(True)
        warn_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(warn_lbl)

        btn_row = QHBoxLayout()

        def do_load():
            dlg.accept()
            threading.Thread(target=self._load_model_thread, daemon=True).start()

        load_btn = QPushButton("Load anyway")
        load_btn.setObjectName("RedBtn")
        load_btn.clicked.connect(do_load)
        btn_row.addWidget(load_btn, 1)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(cancel_btn, 1)
        layout.addLayout(btn_row)

        dlg.exec()

    def _get_model_config(self):
        return {
            "max_initial_tokens": 250,
            "max_context_tokens": 1200,
            "temperature": 0.25,
            "top_p": 0.8,
            "min_response_length": 30,
            "force_context": True,
            "chunk_size": 400,
            "overlap": 80,
        }

    def _load_model_thread(self):
        model_name = Path(self.model_path).stem
        self._run_on_main(lambda: self.load_button.setEnabled(False))
        self._run_on_main(lambda: self.loaded_model_label.setText(f"Loading: {model_name}"))

        loading_done = False

        def fake_progress():
            progress = 0
            while not loading_done:
                progress = min(progress + 1, 95)
                self._run_on_main(lambda p=progress: self.progress_bar.setValue(p))
                self._run_on_main(lambda p=progress: self.update_status(
                    f"Loading {model_name}... {p}%"))
                time.sleep(0.3)

        threading.Thread(target=fake_progress, daemon=True).start()

        try:
            if Llama is None:
                raise RuntimeError("Install llama-cpp-python")

            if self.model is not None:
                self._run_on_main(lambda: self.update_status("Unloading current model..."))
                try:
                    del self.model
                    self.model = None
                    gc.collect()
                    # NVIDIA CUDA cache cleanup
                    if self.gpu_info.get("backend") == "cuda":
                        torch = self._get_torch_module()
                        if torch is not None and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    # Apple Metal: Python gc.collect() is sufficient
                    # Vulkan: no explicit cache API needed
                except Exception:
                    pass

            start = time.time()

            model_filename = Path(self.model_path).name
            per_model = self.model_configs.get(model_filename, {})

            if self.system_ram >= 32:
                target_n_ctx = 8192
            elif self.system_ram >= 16:
                target_n_ctx = 4096
            elif self.system_ram >= 8:
                target_n_ctx = 2048
            else:
                target_n_ctx = 1024
            target_n_ctx = per_model.get("n_ctx", target_n_ctx)

            n_threads = per_model.get("n_threads", self.config["n_threads"])

            self._run_on_main(lambda: self.update_status(
                f"Loading {Path(self.model_path).stem}... (may take a minute)"))

            try:
                self.model = Llama(
                    model_path=self.model_path,
                    n_threads=n_threads,
                    n_gpu_layers=self.config["n_gpu_layers"],
                    n_ctx=target_n_ctx,
                    flash_attn=True,
                    verbose=False,
                )
            except TypeError:
                # Older llama-cpp-python without flash_attn support
                self.model = Llama(
                    model_path=self.model_path,
                    n_threads=n_threads,
                    n_gpu_layers=self.config["n_gpu_layers"],
                    n_ctx=target_n_ctx,
                    verbose=False,
                )

            self.actual_n_ctx = target_n_ctx
            try:
                n = getattr(self.model, "n_ctx", None)
                if callable(n):
                    n = n()
                if isinstance(n, (int, float)) and n > 0:
                    self.actual_n_ctx = int(n)
            except Exception:
                pass

            self.model_config = self._get_model_config()

            model_ctx = None
            if hasattr(self.model, "n_ctx"):
                model_ctx = self.model.n_ctx
            elif hasattr(self.model, "ctx_size"):
                model_ctx = self.model.ctx_size

            if isinstance(model_ctx, (int, float)) and model_ctx > 0:
                if model_ctx <= 512:
                    self.model_config["max_initial_tokens"] = min(
                        self.model_config["max_initial_tokens"], 100)
                    self.model_config["max_context_tokens"] = min(
                        self.model_config["max_context_tokens"], 380)
                elif model_ctx <= 1024:
                    self.model_config["max_initial_tokens"] = min(
                        self.model_config["max_initial_tokens"], 150)
                    self.model_config["max_context_tokens"] = min(
                        self.model_config["max_context_tokens"], 768)
                elif model_ctx <= 2048:
                    self.model_config["max_initial_tokens"] = min(
                        self.model_config["max_initial_tokens"], 220)
                    self.model_config["max_context_tokens"] = min(
                        self.model_config["max_context_tokens"], 1500)

            if "temperature" in per_model:
                self.model_config["temperature"] = per_model["temperature"]

            load_time = round(time.time() - start, 2)
            loading_done = True

            self._run_on_main(lambda: self.progress_bar.setValue(100))
            name = Path(self.model_path).name
            self._run_on_main(lambda n=name: self.loaded_model_label.setText(
                f"Loaded: {n}"))
            self._run_on_main(lambda t=load_time: self.update_status(
                f"Loaded in {t}s"))
            self._run_on_main(lambda: self.send_btn.setEnabled(True))

        except Exception as e:
            loading_done = True
            self._run_on_main(lambda err=str(e): self.update_status(
                f"Error: {err}"))
            self._run_on_main(lambda: self.send_btn.setEnabled(False))

        finally:
            self._run_on_main(lambda: self.load_button.setEnabled(True))

    # ── HuggingFace model downloader ───────────────────────────────

    def _open_hf_downloader(self):
        max_gb = self._get_max_model_size_gb()
        vram = self.gpu_info.get("vram", 0)
        gpu_type = self.gpu_info.get("type", "CPU")
        if gpu_type == "NVIDIA":
            mode_str = f"NVIDIA GPU ({vram}GB VRAM)"
        elif gpu_type == "APPLE_METAL":
            mode_str = f"Apple Metal ({vram}GB unified)"
        elif gpu_type == "AMD":
            mode_str = f"AMD Vulkan ({vram}GB VRAM)"
        else:
            mode_str = "CPU-only"

        dlg = QDialog(self.root)
        dlg.setWindowTitle("⬇ HuggingFace Model Downloader")
        dlg.resize(580, 580)
        layout = QVBoxLayout(dlg)

        info_text = (f"System: {self.system_ram}GB RAM | {mode_str} | "
                     f"Max compatible size: {max_gb}GB  —  ✅ fits  ⚠ too large")
        info_lbl = QLabel(info_text)
        info_lbl.setObjectName("Accent")
        info_lbl.setWordWrap(True)
        layout.addWidget(info_lbl)

        # Search row
        search_row = QHBoxLayout()
        query_entry = QLineEdit()
        query_entry.setPlaceholderText("Search GGUF models… e.g. gemma, qwen, llama")
        search_row.addWidget(query_entry, 1)

        search_btn = QPushButton("🔍 Search")
        search_btn.setObjectName("AccentBtn")
        search_btn.setFixedWidth(100)
        search_row.addWidget(search_btn)
        layout.addLayout(search_row)

        status_lbl = QLabel("")
        layout.addWidget(status_lbl)

        # Results scroll area
        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(4, 4, 4, 4)
        results_layout.setSpacing(4)
        results_layout.addStretch()
        results_scroll.setWidget(results_widget)
        layout.addWidget(results_scroll, 1)

        dl_status_lbl = QLabel("")
        layout.addWidget(dl_status_lbl)
        dl_progress = QProgressBar()
        dl_progress.setRange(0, 100)
        dl_progress.setValue(0)
        dl_progress.setTextVisible(False)
        dl_progress.setFixedHeight(10)
        layout.addWidget(dl_progress)

        active_flag = [False]

        def do_search():
            q = query_entry.text().strip()
            if not q:
                return
            # Clear results
            while results_layout.count() > 1:
                item = results_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            status_lbl.setText("🔍 Searching HuggingFace…")
            threading.Thread(
                target=self._hf_search_thread,
                args=(q, results_layout, status_lbl,
                      dl_status_lbl, dl_progress, active_flag, max_gb, dlg),
                daemon=True,
            ).start()

        search_btn.clicked.connect(do_search)
        query_entry.returnPressed.connect(do_search)

        dlg.exec()

    def _hf_search_thread(self, query, results_layout, status_lbl,
                          dl_status_lbl, dl_progress, active_flag, max_gb, dlg):
        try:
            url = (f"https://huggingface.co/api/models"
                   f"?search={quote(query)}&filter=gguf"
                   f"&sort=downloads&direction=-1&limit=20&full=true")
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            models = resp.json()
        except Exception as e:
            self._run_on_main(lambda: status_lbl.setText(
                f"❌ Search failed: {e}"))
            return

        rows = []
        for m in models:
            model_id = m.get("modelId") or m.get("id", "")
            for sib in m.get("siblings", []):
                fname = sib.get("rfilename", "")
                if not fname.lower().endswith(".gguf"):
                    continue
                size_bytes = None
                lfs = sib.get("lfs")
                if isinstance(lfs, dict):
                    size_bytes = lfs.get("size")
                if size_bytes is None:
                    size_bytes = sib.get("size")
                size_gb = round(size_bytes / (1024 ** 3), 2) if size_bytes else None
                compatible = size_gb is not None and size_gb <= max_gb
                unknown = size_gb is None
                rows.append((model_id, fname, size_gb, compatible, unknown))

        if not rows:
            self._run_on_main(lambda: status_lbl.setText(
                "No GGUF files found. Try a broader search term."))
            return

        rows.sort(key=lambda r: (0 if (r[3] or r[4]) else 1, r[2] or 999))
        self._run_on_main(lambda: status_lbl.setText(
            f"Found {len(rows)} GGUF file(s) — compatible ones listed first"))
        self._run_on_main(lambda: self._build_hf_results(
            rows, results_layout, dl_status_lbl, dl_progress, active_flag, dlg))

    def _build_hf_results(self, rows, results_layout, dl_status_lbl,
                          dl_progress, active_flag, dlg):
        # Clear existing
        while results_layout.count() > 1:
            item = results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for model_id, fname, size_gb, compatible, unknown in rows:
            if unknown:
                badge = "❓"
            elif compatible:
                badge = "✅"
            else:
                badge = "⚠"

            size_str = f"{size_gb}GB" if size_gb is not None else "size unknown"

            card = QFrame()
            card.setFrameShape(QFrame.StyledPanel)
            card_layout = QHBoxLayout(card)
            card_layout.setContentsMargins(10, 6, 10, 6)

            info_col = QVBoxLayout()
            name_lbl = QLabel(f"{badge} {model_id}")
            name_lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
            name_lbl.setWordWrap(True)
            info_col.addWidget(name_lbl)
            file_lbl = QLabel(f"  {fname}   {size_str}")
            file_lbl.setObjectName("Muted")
            info_col.addWidget(file_lbl)
            card_layout.addLayout(info_col, 1)

            if compatible or unknown:
                def make_dl_cmd(mid=model_id, fn=fname):
                    def cmd():
                        if active_flag[0]:
                            return
                        active_flag[0] = True
                        dl_progress.setValue(0)
                        threading.Thread(
                            target=self._hf_download_thread,
                            args=(mid, fn, dl_status_lbl,
                                  dl_progress, active_flag),
                            daemon=True,
                        ).start()
                    return cmd

                dl_btn = QPushButton("⬇ Download")
                dl_btn.setObjectName("AccentBtn")
                dl_btn.setFixedWidth(100)
                dl_btn.clicked.connect(make_dl_cmd())
                card_layout.addWidget(dl_btn)
            else:
                warn_lbl = QLabel("Too large")
                warn_lbl.setStyleSheet("color: #F38BA8;")
                card_layout.addWidget(warn_lbl)

            # Insert before stretch
            count = results_layout.count()
            results_layout.insertWidget(count - 1, card)

    def _hf_download_thread(self, model_id, filename, status_lbl,
                            progress_bar, active_flag):
        dest_dir = Path(app_data_path("models"))
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / filename

        def upd_status(t):
            self._run_on_main(lambda: status_lbl.setText(t))

        def upd_progress(v):
            self._run_on_main(lambda: progress_bar.setValue(int(v * 100)))

        try:
            dl_url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"
            upd_status(f"⬇ Connecting to {model_id}…")
            with requests.get(dl_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                downloaded = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                upd_progress(downloaded / total)
                                mb = downloaded / (1024 * 1024)
                                total_mb = total / (1024 * 1024)
                                upd_status(
                                    f"⬇ {filename}: {mb:.0f}/{total_mb:.0f} MB")
            upd_progress(1.0)
            upd_status(f"✅ Downloaded: {filename} — ready to load!")
            self._run_on_main(self._scan_models_on_startup)
        except Exception as e:
            upd_status(f"❌ Download failed: {e}")
            if dest.exists():
                try:
                    dest.unlink()
                except Exception:
                    pass
        finally:
            active_flag[0] = False

    # ── GPU detection / system monitor ─────────────────────────────

    def detect_gpu(self):
        torch = self._get_torch_module()
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    return "NVIDIA"
            except Exception:
                pass
        # Check detected GPU info from startup
        gpu_type = getattr(self, 'gpu_info', {}).get('type', 'CPU')
        if gpu_type in ('NVIDIA', 'AMD', 'APPLE_METAL'):
            return gpu_type
        try:
            from llama_cpp import llama_cpp  # noqa: F401
            return "GPU_POSSIBLE"
        except Exception:
            pass
        return "CPU"

    def detect_gpu_info(self):
        # ── NVIDIA (CUDA) ──────────────────────────────────────
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                vram_mb = int(result.stdout.strip().splitlines()[0].strip())
                return {"type": "NVIDIA", "backend": "cuda",
                        "vram": round(vram_mb / 1024, 2)}
        except Exception:
            pass

        # ── Apple Metal (macOS) ────────────────────────────────
        if sys.platform == "darwin":
            try:
                import platform as _plat
                machine = _plat.machine().lower()
                # Apple Silicon (M1/M2/M3/M4) — unified memory = system RAM
                if "arm" in machine or "aarch64" in machine:
                    total_ram = round(
                        psutil.virtual_memory().total / (1024**3), 1)
                    return {"type": "APPLE_METAL", "backend": "metal",
                            "vram": total_ram}
                # Intel Mac — check for Metal-capable GPU via system_profiler
                sp = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True, text=True, timeout=5,
                )
                if sp.returncode == 0 and "Metal" in sp.stdout:
                    # Estimate VRAM from system_profiler output
                    vram_match = re.search(
                        r"VRAM.*?:\s*(\d+)\s*(MB|GB)", sp.stdout, re.I)
                    if vram_match:
                        v = int(vram_match.group(1))
                        if vram_match.group(2).upper() == "MB":
                            v = round(v / 1024, 1)
                        return {"type": "APPLE_METAL", "backend": "metal",
                                "vram": v}
                    return {"type": "APPLE_METAL", "backend": "metal",
                            "vram": 0}
            except Exception:
                pass

        # ── AMD / Intel (Vulkan) ───────────────────────────────
        try:
            # Windows: check for AMD GPU via WMIC
            if sys.platform == "win32":
                wmic = subprocess.run(
                    ["wmic", "path", "win32_videocontroller", "get",
                     "Name,AdapterRAM", "/format:csv"],
                    capture_output=True, text=True, timeout=5,
                )
                if wmic.returncode == 0:
                    for line in wmic.stdout.strip().splitlines():
                        parts = line.strip().split(",")
                        if len(parts) >= 3:
                            adapter_ram = parts[1].strip()
                            name = parts[2].strip().lower()
                            if "amd" in name or "radeon" in name:
                                try:
                                    vram_bytes = int(adapter_ram)
                                    vram_gb = round(
                                        vram_bytes / (1024**3), 1)
                                    return {
                                        "type": "AMD",
                                        "backend": "vulkan",
                                        "vram": max(vram_gb, 0),
                                    }
                                except (ValueError, TypeError):
                                    return {"type": "AMD",
                                            "backend": "vulkan",
                                            "vram": 0}
            else:
                # Linux: check lspci for AMD GPU
                lspci = subprocess.run(
                    ["lspci"], capture_output=True, text=True, timeout=5)
                if lspci.returncode == 0:
                    for line in lspci.stdout.splitlines():
                        ll = line.lower()
                        if ("vga" in ll or "3d" in ll or "display" in ll):
                            if "amd" in ll or "radeon" in ll:
                                return {"type": "AMD",
                                        "backend": "vulkan", "vram": 0}
        except Exception:
            pass

        return {"type": "CPU", "backend": "cpu", "vram": 0}

    def _detect_llama_cpp_backend(self):
        """Detect which GPU backend the installed llama-cpp-python was
        compiled with. Returns 'cuda', 'metal', 'vulkan', or 'cpu'."""
        if Llama is None:
            return "cpu"
        try:
            from llama_cpp import llama_cpp as _ll
            # Check for Metal support (macOS)
            if hasattr(_ll, "llama_supports_metal"):
                if _ll.llama_supports_metal():
                    return "metal"
            # Check for GPU offload support (generic)
            if hasattr(_ll, "llama_supports_gpu_offload"):
                if _ll.llama_supports_gpu_offload():
                    # Determine which backend based on OS + detected GPU
                    if sys.platform == "darwin":
                        return "metal"
                    gpu_type = self.gpu_info.get("type", "CPU")
                    if gpu_type == "NVIDIA":
                        return "cuda"
                    if gpu_type == "AMD":
                        return "vulkan"
                    return "cuda"  # best guess
        except Exception:
            pass
        # Fallback: check if n_gpu_layers > 0 works without error
        return "cpu"

    def _get_gpu_install_command(self):
        """Return the pip command to reinstall llama-cpp-python with the
        correct GPU backend for this system."""
        gpu_type = self.gpu_info.get("type", "CPU")
        backend = self.gpu_info.get("backend", "cpu")
        py = sys.executable

        base = (f'"{py}" -m pip install llama-cpp-python --force-reinstall '
                f'--no-cache-dir')

        if gpu_type == "NVIDIA":
            return (f'set CMAKE_ARGS="-DGGML_CUDA=on" && {base}',
                    "CUDA (NVIDIA)")
        elif backend == "metal":
            return (f'CMAKE_ARGS="-DGGML_METAL=on" {base}',
                    "Metal (Apple GPU)")
        elif gpu_type == "AMD" or backend == "vulkan":
            if sys.platform == "win32":
                return (f'set CMAKE_ARGS="-DGGML_VULKAN=on" && {base}',
                        "Vulkan (AMD)")
            else:
                return (f'CMAKE_ARGS="-DGGML_VULKAN=on" {base}',
                        "Vulkan (AMD)")
        return (base, "CPU-only")

    def start_system_monitor(self):
        def update():
            while not getattr(self, '_shutting_down', False):
                cpu = psutil.cpu_percent()
                ram = psutil.virtual_memory().percent
                gpu_text = ""
                gpu_type = self.gpu_info.get("type", "CPU")
                gpu_backend = self.gpu_info.get("backend", "cpu")

                if gpu_type == "NVIDIA":
                    # NVIDIA: use torch CUDA utilization or nvidia-smi
                    torch = self._get_torch_module()
                    if torch is not None:
                        try:
                            if torch.cuda.is_available():
                                gpu = (torch.cuda.utilization(0)
                                       if hasattr(torch.cuda, "utilization")
                                       else 0)
                                gpu_text = f" | GPU: {gpu}%"
                        except Exception:
                            pass
                    if not gpu_text:
                        try:
                            r = subprocess.run(
                                ["nvidia-smi",
                                 "--query-gpu=utilization.gpu",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=3,
                            )
                            if r.returncode == 0 and r.stdout.strip():
                                gpu_text = f" | GPU: {r.stdout.strip()}%"
                        except Exception:
                            pass

                elif gpu_type == "APPLE_METAL":
                    # macOS: use powermetrics GPU usage (if available)
                    # Lightweight fallback: just show "Metal" tag
                    gpu_text = " | Metal"
                    try:
                        r = subprocess.run(
                            ["ioreg", "-r", "-d", "1", "-c",
                             "IOAccelerator"],
                            capture_output=True, text=True, timeout=3,
                        )
                        if r.returncode == 0 and "PerformanceState" in r.stdout:
                            gpu_text = " | Metal ✓"
                    except Exception:
                        pass

                elif gpu_type == "AMD":
                    # AMD Vulkan: try rocm-smi on Linux, basic tag otherwise
                    gpu_text = " | Vulkan"
                    if sys.platform != "win32":
                        try:
                            r = subprocess.run(
                                ["rocm-smi", "--showuse",
                                 "--csv"],
                                capture_output=True, text=True, timeout=3,
                            )
                            if r.returncode == 0:
                                lines = r.stdout.strip().splitlines()
                                if len(lines) >= 2:
                                    val = lines[1].split(",")[0].strip()
                                    gpu_text = f" | GPU: {val}%"
                        except Exception:
                            pass

                text = f"CPU: {cpu}% | RAM: {ram}%{gpu_text}"
                if not getattr(self, '_shutting_down', False):
                    self.signals.run_on_main.emit(
                        lambda t=text: self.system_monitor_label.setText(t))
                time.sleep(2)

        threading.Thread(target=update, daemon=True).start()
