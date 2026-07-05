"""Standalone helper classes — SystemTools and TokenOptimizer."""

import json
import os
import platform
import shutil
import time
from typing import Dict


class SystemTools:
    @staticmethod
    def list_files(path: str) -> str:
        try:
            entries = sorted(os.listdir(path))
            return json.dumps(entries)
        except Exception as exc:
            return f"ERROR: {str(exc)}"

    @staticmethod
    def get_disk_space() -> str:
        if platform.system() == "Windows":
            total, used, free = shutil.disk_usage("C:\\")
        else:
            total, used, free = shutil.disk_usage("/")
        return json.dumps({
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
        })

    @staticmethod
    def get_system_uptime() -> str:
        if platform.system() == "Windows":
            uptime_seconds = time.perf_counter()
        else:
            with open("/proc/uptime", "r") as f:
                uptime_seconds = float(f.readline().split()[0])
        return json.dumps({"uptime_seconds": round(uptime_seconds, 2)})


class TokenOptimizer:
    """Optimize tokens for small-context models using quantization-inspired techniques"""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count (conservative: assume 1 token per word + 20% overhead)"""
        if not text:
            return 0
        word_count = len(text.split())
        return int(word_count * 1.2)

    @staticmethod
    def compress_context(context: str, max_chars: int = 200) -> str:
        """Compress context using extractive summarization (most important sentences)."""
        if not context or len(context) <= max_chars:
            return context

        sentences = context.replace(".", ".\n").replace("!", "!\n").replace("?", "?\n").split("\n")
        sentences = [s.strip() for s in sentences if s.strip()]

        scored = []
        for i, sent in enumerate(sentences):
            length_score = 1 if 20 < len(sent) < 100 else 0.5
            position_score = 1 - (i / len(sentences)) * 0.3
            score = length_score * position_score
            scored.append((sent, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in scored[:3]]
        result = " ".join(selected)

        return result[:max_chars]

    @staticmethod
    def optimize_for_small_context(context: str, query: str, max_tokens: int = 100) -> str:
        """Optimize context for small-context models (512-1024 tokens)."""
        if not context:
            return ""

        max_chars = max_tokens * 4

        if len(context) <= max_chars:
            return context

        query_keywords = set(w.lower() for w in query.split() if len(w) > 3)

        lines = context.split("\n")
        scored_lines = []

        for line in lines:
            if not line.strip():
                continue
            match_count = sum(1 for kw in query_keywords if kw in line.lower())
            score = match_count + (1 if len(line) > 20 else 0)
            scored_lines.append((line.strip(), score))

        scored_lines.sort(key=lambda x: x[1], reverse=True)
        selected_lines = [line for line, _ in scored_lines[:5]]

        result = "\n".join(selected_lines)
        return result[:max_chars]

    @staticmethod
    def build_token_budget(model_context_window: int) -> Dict:
        """Calculate token budget for small-context models"""
        return {
            "total_window": model_context_window,
            "prompt_overhead": 20,
            "min_response": 30,
            "max_context": max(50, model_context_window - 20 - 30),
            "max_response": min(100, model_context_window - 20 - 30),
        }
