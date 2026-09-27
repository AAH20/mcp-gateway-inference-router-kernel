"""
Unit tests for integrated McpGatewayInferenceRouterEngine.
"""

import unittest
from mcp_gateway_inference_router_kernel.engine import McpGatewayInferenceRouterEngine


class TestMcpGatewayEngine(unittest.TestCase):

    def setUp(self):
        self.engine = McpGatewayInferenceRouterEngine()

    def test_run_schema_pruning_benchmark(self):
        res = self.engine.run_schema_pruning_benchmark(num_tools=50)
        self.assertIn("tokens_saved", res)
        self.assertGreater(res["tokens_saved"], 0)

    def test_run_radix_cache_benchmark(self):
        res = self.engine.run_radix_cache_benchmark(num_requests=50)
        self.assertIn("cache_hit_requests_pct", res)
        self.assertGreater(res["cache_hit_requests_pct"], 0.0)

    def test_run_model_cascading_benchmark(self):
        res = self.engine.run_model_cascading_benchmark(num_queries=30)
        self.assertIn("blended_savings_pct", res)
        self.assertGreater(res["blended_savings_pct"], 0.0)

    def test_run_deadlock_orchestration_benchmark(self):
        res = self.engine.run_deadlock_orchestration_benchmark()
        self.assertIn("deadlocks_prevented", res)
        self.assertEqual(res["total_tasks_scheduled"], 4)
        self.assertGreater(res["total_makespan_ms"], 0.0)

    def test_run_context_distillation_benchmark(self):
        res = self.engine.run_context_distillation_benchmark()
        self.assertIn("compression_ratio_pct", res)
        self.assertGreater(res["compression_ratio_pct"], 0.0)

    def test_run_water_filling_benchmark(self):
        res = self.engine.run_water_filling_benchmark(num_burst_requests=30)
        self.assertIn("guaranteed_delivery_pct", res)
        self.assertGreater(res["guaranteed_delivery_pct"], 90.0)

    def test_run_comprehensive_benchmark(self):
        rep = self.engine.run_comprehensive_benchmark()
        self.assertGreater(rep.total_runtime_ms, 0.0)
        self.assertIsNotNone(rep.schema_pruning_summary)
        self.assertIsNotNone(rep.radix_cache_summary)
        self.assertIsNotNone(rep.model_cascading_summary)
        self.assertIsNotNone(rep.deadlock_orchestration_summary)
        self.assertIsNotNone(rep.context_distillation_summary)
        self.assertIsNotNone(rep.water_filling_summary)


if __name__ == "__main__":
    unittest.main()
