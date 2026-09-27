"""
Unit tests for Multi-Provider Water-Filling Router.
"""

import unittest
from mcp_gateway_inference_router_kernel.core.models import ProviderEndpoint
from mcp_gateway_inference_router_kernel.core.multi_provider_water_filling import MultiProviderWaterFillingRouter


class TestMultiProviderWaterFilling(unittest.TestCase):

    def setUp(self):
        self.providers = [
            ProviderEndpoint("p1", "Provider A", rpm_limit=100, tpm_limit=5000, current_rpm=0, current_tpm=0, error_rate_429=0.0, cost_per_1k_tokens=0.002),
            ProviderEndpoint("p2", "Provider B", rpm_limit=100, tpm_limit=5000, current_rpm=0, current_tpm=0, error_rate_429=0.0, cost_per_1k_tokens=0.002),
        ]
        self.router = MultiProviderWaterFillingRouter(self.providers)

    def test_load_balanced_water_filling(self):
        # Dispatch 2 requests of 2000 tokens each: should balance between p1 and p2
        alloc1 = self.router.allocate_request(2000)
        alloc2 = self.router.allocate_request(2000)

        self.assertTrue(alloc1.guaranteed_no_429)
        self.assertTrue(alloc2.guaranteed_no_429)
        self.assertNotEqual(alloc1.allocated_provider_id, alloc2.allocated_provider_id)


if __name__ == "__main__":
    unittest.main()
