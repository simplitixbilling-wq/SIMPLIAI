import unittest

from app_core.runtime_planner import RuntimePlanner, looks_incomplete


class RuntimePlannerTests(unittest.TestCase):
    def test_compact_plan_uses_resumable_small_passes(self):
        plan = RuntimePlanner(system_ram_gb=8, gpu_info={}).plan(4096)

        self.assertEqual(plan.tier, "compact")
        self.assertTrue(plan.continuation_enabled)
        self.assertGreaterEqual(plan.max_continuations, 2)
        self.assertLessEqual(plan.response_token_ceiling, 768)
        self.assertEqual(plan.ocr_parallelism, 1)

    def test_high_ram_plan_keeps_deeper_retrieval(self):
        plan = RuntimePlanner(system_ram_gb=64, gpu_info={"vram": 16}).plan(32768)

        self.assertEqual(plan.tier, "max")
        self.assertGreaterEqual(plan.rag_top_k, 12)
        self.assertGreater(plan.response_token_ceiling, 1500)

    def test_response_budget_respects_floor_ceiling_and_user_cap(self):
        plan = RuntimePlanner(system_ram_gb=8, gpu_info={}).plan(4096)

        self.assertEqual(plan.response_budget(5000), plan.response_token_ceiling)
        self.assertEqual(plan.response_budget(5000, user_max_tokens=300), 300)
        self.assertGreaterEqual(plan.response_budget(100), 64)

    def test_incomplete_detection_only_near_budget(self):
        self.assertFalse(looks_incomplete("This is complete.", token_count=50, token_budget=100))
        self.assertTrue(looks_incomplete("This answer continues with", token_count=96, token_budget=100))
        self.assertFalse(looks_incomplete("This answer is complete.", token_count=96, token_budget=100))


if __name__ == "__main__":
    unittest.main()
