"""
Unit tests for Conformal Model Cascader.
"""

import unittest
from mcp_gateway_inference_router_kernel.core.models import (
    ModelTierClass,
    QueryCharacteristics,
)
from mcp_gateway_inference_router_kernel.core.conformal_model_cascader import ConformalModelCascader


class TestConformalModelCascader(unittest.TestCase):

    def setUp(self):
        self.cascader = ConformalModelCascader()

    def test_simple_query_routes_to_cheap_tier(self):
        simple_q = QueryCharacteristics(
            query_id="simple_1",
            token_length=150,
            code_density=0.05,
            reasoning_depth_est=0.1,
            required_tool_count=1,
            max_budget_usd=0.01,
            latency_deadline_ms=500.0,
        )
        dec = self.cascader.route_query(simple_q)
        self.assertEqual(dec.routed_tier, ModelTierClass.CHEAP_FAST)
        self.assertGreater(dec.cost_savings_pct, 50.0)

    def test_complex_query_routes_to_frontier_tier(self):
        complex_q = QueryCharacteristics(
            query_id="complex_1",
            token_length=7000,
            code_density=0.95,
            reasoning_depth_est=0.90,
            required_tool_count=8,
            max_budget_usd=0.10,
            latency_deadline_ms=5000.0,
        )
        dec = self.cascader.route_query(complex_q)
        self.assertEqual(dec.routed_tier, ModelTierClass.FRONTIER_REASONER)


if __name__ == "__main__":
    unittest.main()
