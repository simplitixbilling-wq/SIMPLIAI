"""
Memory Optimizer — Virtual RAM & Adaptive Performance for SIMPLE_AI
===================================================================
Uses SSD-backed page file as virtual RAM extension via mmap, so models
can run with FULL (or BIGGER) context even when physical RAM is limited.

Key ideas (systems engineering approach):
  1. mmap=True  → model weights live on disk, OS pages them into RAM on demand.
                   On SSD this is nearly as fast as physical RAM — effectively
                   EXTENDS your usable memory by the size of the page file.
  2. use_mlock=False → allow OS to swap model pages to SSD under pressure,
                        freeing physical RAM for the KV cache and inference.
  3. n_batch tuning → smaller batches reduce peak memory SPIKES without
                      reducing context or quality. Only change under pressure.
  4. Context & threads are NEVER reduced — the whole point is to let the
     model run at full power using virtual memory as overflow.
  5. Background watchdog → monitors RAM, emits info to UI, runs GC proactively.
  6. Page file advisor → checks Windows virtual memory config and recommends fixes.
"""

import os
import sys
import gc
import time
import threading
import logging
import ctypes
import psutil

log = logging.getLogger("memory_optimizer")

# ── Pressure stages ────────────────────────────────────────────
# n_batch is the ONLY thing reduced. Context & threads stay full.

PRESSURE_STAGES = {
    "normal":   {"avail_gb_above": 4.0, "n_batch": 512},
    "moderate": {"avail_gb_above": 2.5, "n_batch": 256},
    "high":     {"avail_gb_above": 1.5, "n_batch": 128},
    "critical": {"avail_gb_above": 0.0, "n_batch":  64},
}
STAGE_ORDER = ["normal", "moderate", "high", "critical"]


class MemoryOptimizer:
    """Enables SSD-backed virtual RAM for AI models via mmap.
    
    NEVER reduces n_ctx or threads — those stay at full user-requested values.
    Only tunes n_batch (peak spike control) and enables mmap/use_mlock settings.
    """

    def __init__(self, system_ram_gb: int = 0, gpu_info: dict = None, emit_fn=None):
        self.system_ram = system_ram_gb or round(psutil.virtual_memory().total / (1024**3))
        self.gpu_info = gpu_info or {}
        self._emit = emit_fn  # optional callback for UI warnings
        self._stage = "normal"
        self._lock = threading.Lock()
        self._watchdog_running = False
        self._peak_rss_mb = 0
        self._pagefile_info = None

    # ── Public API ─────────────────────────────────────────────

    @property
    def stage(self) -> str:
        with self._lock:
            return self._stage

    def snapshot(self) -> dict:
        """Current memory state."""
        vm = psutil.virtual_memory()
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss / (1024**2)
        self._peak_rss_mb = max(self._peak_rss_mb, rss)
        return {
            "total_gb": round(vm.total / (1024**3), 1),
            "available_gb": round(vm.available / (1024**3), 2),
            "used_percent": vm.percent,
            "process_rss_mb": round(rss, 1),
            "peak_rss_mb": round(self._peak_rss_mb, 1),
            "stage": self.stage,
            "swap_total_gb": round(psutil.swap_memory().total / (1024**3), 1),
            "swap_used_gb": round(psutil.swap_memory().used / (1024**3), 1),
        }

    def optimal_llama_params(self, base_n_ctx: int, base_n_threads: int,
                             base_n_gpu_layers: int) -> dict:
        """Return llama.cpp kwargs with mmap enabled for SSD virtual RAM.

        Context window and threads are NEVER reduced — they stay at full
        user-requested values. Only n_batch is tuned to control peak spikes,
        and mmap + use_mlock are set to let the OS use SSD as overflow.
        """
        stage = self._evaluate_stage()
        cfg = PRESSURE_STAGES[stage]

        params = {
            "n_ctx": base_n_ctx,             # FULL context — never reduced
            "n_threads": base_n_threads,     # FULL threads — never reduced
            "n_gpu_layers": base_n_gpu_layers,
            "n_batch": cfg["n_batch"],       # only this adapts under pressure
            "use_mlock": False,              # let OS swap model pages to SSD
            "verbose": False,
        }

        if stage != "normal":
            log.info(
                "Memory stage [%s]: mmap enabled, n_batch=%d (ctx=%d, threads=%d kept full)",
                stage, cfg["n_batch"], base_n_ctx, base_n_threads)

        return params

    def optimal_inference_params(self, base_max_tokens: int) -> dict:
        """Return inference params — max_tokens is NEVER reduced.
        Only n_batch adapts to smooth out memory spikes during generation."""
        stage = self._evaluate_stage()
        cfg = PRESSURE_STAGES[stage]
        return {"max_tokens": base_max_tokens, "n_batch": cfg["n_batch"], "stage": stage}

    def force_gc(self):
        """Aggressive garbage collection."""
        gc.collect()
        gc.collect()
        # Attempt to trim process working set on Windows
        if sys.platform == "win32":
            try:
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetCurrentProcess()
                kernel32.SetProcessWorkingSetSize(handle, -1, -1)
            except Exception:
                pass

    def pagefile_advice(self) -> dict:
        """Check Windows page file (swap) config and return recommendations."""
        info = {"ok": True, "messages": [], "swap_total_gb": 0, "on_ssd": None}
        try:
            swap = psutil.swap_memory()
            info["swap_total_gb"] = round(swap.total / (1024**3), 1)

            # Recommend: page file >= 1.5x physical RAM for AI workloads
            recommended = round(self.system_ram * 1.5, 1)

            if info["swap_total_gb"] < 2:
                info["ok"] = False
                info["messages"].append(
                    f"Page file is only {info['swap_total_gb']}GB. "
                    f"Recommended: {recommended}GB minimum for AI models.")
                info["messages"].append(
                    "How to fix: Settings → System → About → Advanced → "
                    "Performance → Virtual Memory → set Custom size on your SSD.")
            elif info["swap_total_gb"] < recommended:
                info["messages"].append(
                    f"Page file is {info['swap_total_gb']}GB. "
                    f"Ideal: {recommended}GB on an SSD for best AI performance.")
            else:
                info["messages"].append(
                    f"Page file is {info['swap_total_gb']}GB — good for AI workloads.")

            # Check if system drive is SSD (heuristic)
            if sys.platform == "win32":
                info["on_ssd"] = self._check_ssd()
                if info["on_ssd"] is False:
                    info["ok"] = False
                    info["messages"].append(
                        "WARNING: Page file appears to be on HDD. "
                        "Move it to an SSD for 10-50x faster virtual memory.")
                elif info["on_ssd"] is True:
                    info["messages"].append("Page file is on SSD — good.")
        except Exception as e:
            info["messages"].append(f"Could not check page file: {e}")

        self._pagefile_info = info
        return info

    def estimate_model_ram(self, model_size_gb: float, n_ctx: int,
                           n_layers: int = 24, d_model: int = 1536) -> dict:
        """Estimate total RAM needed for a model + KV cache."""
        # KV cache ≈ 2 * n_layers * n_ctx * d_model * 2 bytes (fp16)
        kv_cache_bytes = 2 * n_layers * n_ctx * d_model * 2
        kv_cache_gb = kv_cache_bytes / (1024**3)

        # Working memory overhead (activations, scratch buffers)
        overhead_gb = 0.3 + (model_size_gb * 0.1)

        total = model_size_gb + kv_cache_gb + overhead_gb
        return {
            "model_gb": round(model_size_gb, 2),
            "kv_cache_gb": round(kv_cache_gb, 2),
            "overhead_gb": round(overhead_gb, 2),
            "total_gb": round(total, 2),
            "fits_in_ram": total < (self.system_ram * 0.8),
            "needs_virtual": total > (self.system_ram * 0.8),
        }

    # ── Watchdog ───────────────────────────────────────────────

    def start_watchdog(self, interval: float = 3.0):
        """Background thread that monitors memory and emits warnings."""
        if self._watchdog_running:
            return
        self._watchdog_running = True

        def _loop():
            last_stage = "normal"
            while self._watchdog_running:
                try:
                    stage = self._evaluate_stage()
                    snap = self.snapshot()

                    if stage != last_stage:
                        msg = self._stage_message(stage, snap)
                        log.info("Memory stage: %s → %s", last_stage, stage)
                        if self._emit:
                            self._emit("memory_warning", {
                                "stage": stage,
                                "message": msg,
                                "available_gb": snap["available_gb"],
                                "used_percent": snap["used_percent"],
                            })

                        # Auto garbage collect on escalation
                        if STAGE_ORDER.index(stage) > STAGE_ORDER.index(last_stage):
                            self.force_gc()

                        last_stage = stage

                except Exception as e:
                    log.debug("Watchdog tick error: %s", e)

                time.sleep(interval)

        t = threading.Thread(target=_loop, daemon=True, name="mem-watchdog")
        t.start()

    def stop_watchdog(self):
        self._watchdog_running = False

    # ── Internal ───────────────────────────────────────────────

    def _evaluate_stage(self) -> str:
        avail_gb = psutil.virtual_memory().available / (1024**3)
        stage = "critical"
        for s in STAGE_ORDER:
            threshold = PRESSURE_STAGES[s]["avail_gb_above"]
            if avail_gb >= threshold:
                stage = s
                break
        with self._lock:
            self._stage = stage
        return stage

    def _stage_message(self, stage: str, snap: dict) -> str:
        avail = snap["available_gb"]
        if stage == "normal":
            return f"Memory OK — {avail:.1f}GB available"
        elif stage == "moderate":
            return (f"Memory moderate ({avail:.1f}GB free). "
                    "Reducing batch size for stability.")
        elif stage == "high":
            return (f"Low memory ({avail:.1f}GB free). "
                    "Context window and batch size reduced. "
                    "Consider closing other apps or using a smaller model.")
        else:
            return (f"CRITICAL memory ({avail:.1f}GB free). "
                    "Heavily throttled. Close other apps, "
                    "or the model may need to be unloaded.")

    @staticmethod
    def _check_ssd() -> bool | None:
        """Heuristic: check if the Windows system drive is SSD."""
        try:
            import subprocess
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-PhysicalDisk | Where-Object DeviceId -eq 0 "
                 "| Select-Object -ExpandProperty MediaType"],
                capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                media = r.stdout.strip().lower()
                if "ssd" in media or "solid" in media:
                    return True
                elif "hdd" in media or "rotat" in media:
                    return False
        except Exception:
            pass
        return None
