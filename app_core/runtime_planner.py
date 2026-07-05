"""Resource-aware planning for offline AI workflows.

The planner adapts budgets and execution strategy to the machine without
reducing the user-facing task. Low-resource systems use smaller resumable
passes, not abrupt truncation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimePlan:
    tier: str
    system_ram_gb: float
    effective_n_ctx: int
    rag_top_k: int
    rag_context_ratio: float
    file_context_ratio: float
    response_token_floor: int
    response_token_ceiling: int
    continuation_enabled: bool
    max_continuations: int
    ocr_parallelism: int
    batch_rows: int

    def response_budget(self, available_tokens: int, user_max_tokens: int | None = None) -> int:
        budget = max(self.response_token_floor, min(int(available_tokens), self.response_token_ceiling))
        if user_max_tokens and int(user_max_tokens) > 0:
            budget = min(budget, int(user_max_tokens))
        return max(64, budget)

    def rag_budget_chars(self) -> int:
        return max(1200, int(self.effective_n_ctx * 3.5 * self.rag_context_ratio))

    def file_budget_chars(self) -> int:
        return max(1800, int(self.effective_n_ctx * 3.5 * self.file_context_ratio))


class RuntimePlanner:
    """Choose workflow budgets from RAM/GPU/context while preserving capability."""

    def __init__(self, system_ram_gb: float = 0, gpu_info: dict | None = None):
        self.system_ram_gb = float(system_ram_gb or 0)
        self.gpu_info = gpu_info or {}

    def tier(self) -> str:
        ram = self.system_ram_gb
        vram = float((self.gpu_info or {}).get("vram", 0) or 0)
        if ram <= 8 and vram < 6:
            return "compact"
        if ram <= 16 and vram < 8:
            return "balanced"
        if ram <= 32:
            return "power"
        return "max"

    def plan(self, actual_n_ctx: int) -> RuntimePlan:
        n_ctx = max(512, int(actual_n_ctx or 4096))
        tier = self.tier()

        if tier == "compact":
            return RuntimePlan(
                tier=tier,
                system_ram_gb=self.system_ram_gb,
                effective_n_ctx=n_ctx,
                rag_top_k=4,
                rag_context_ratio=0.20,
                file_context_ratio=0.22,
                response_token_floor=384,
                response_token_ceiling=768,
                continuation_enabled=True,
                max_continuations=3,
                ocr_parallelism=1,
                batch_rows=2500,
            )
        if tier == "balanced":
            return RuntimePlan(
                tier=tier,
                system_ram_gb=self.system_ram_gb,
                effective_n_ctx=n_ctx,
                rag_top_k=6,
                rag_context_ratio=0.25,
                file_context_ratio=0.28,
                response_token_floor=512,
                response_token_ceiling=1024,
                continuation_enabled=True,
                max_continuations=2,
                ocr_parallelism=2,
                batch_rows=7500,
            )
        if tier == "power":
            return RuntimePlan(
                tier=tier,
                system_ram_gb=self.system_ram_gb,
                effective_n_ctx=n_ctx,
                rag_top_k=8,
                rag_context_ratio=0.30,
                file_context_ratio=0.35,
                response_token_floor=768,
                response_token_ceiling=1536,
                continuation_enabled=True,
                max_continuations=1,
                ocr_parallelism=4,
                batch_rows=20000,
            )
        return RuntimePlan(
            tier=tier,
            system_ram_gb=self.system_ram_gb,
            effective_n_ctx=n_ctx,
            rag_top_k=12,
            rag_context_ratio=0.35,
            file_context_ratio=0.40,
            response_token_floor=1024,
            response_token_ceiling=2048,
            continuation_enabled=True,
            max_continuations=1,
            ocr_parallelism=6,
            batch_rows=50000,
        )


def looks_incomplete(text: str, token_count: int, token_budget: int) -> bool:
    """Heuristic for whether a generation likely stopped due to budget."""
    if token_budget <= 0 or token_count < max(32, int(token_budget * 0.92)):
        return False
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if stripped.endswith((".", "!", "?", "```", "]", ")", "}")):
        return False
    if stripped.lower().endswith(("in conclusion", "therefore", "because", "and", "or", "the", "with", "for")):
        return True
    return True
