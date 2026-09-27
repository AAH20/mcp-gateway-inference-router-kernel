"""
Unit tests for Radix-Tree Prefix Cache Router.
"""

import unittest
from mcp_gateway_inference_router_kernel.core.models import (
    InferenceWorker,
    RadixRouteRequest,
)
from mcp_gateway_inference_router_kernel.core.radix_prefix_cache_router import RadixPrefixCacheRouter


class TestRadixPrefixCacheRouter(unittest.TestCase):

    def setUp(self):
        self.workers = [
            InferenceWorker("w1", "10.0.0.1", kv_capacity_tokens=10000, active_requests=0),
            InferenceWorker("w2", "10.0.0.2", kv_capacity_tokens=10000, active_requests=0),
        ]
        self.router = RadixPrefixCacheRouter(self.workers)

    def test_repeated_prefix_routes_to_same_worker(self):
        # Realistic prompt with 400 shared prefix tokens
        common_prefix = [f"sys_prompt_{i}" for i in range(400)]
        req1 = RadixRouteRequest("req1", common_prefix + ["user_1"], max_tokens=50)
        dec1 = self.router.route_request(req1)

        # First request has 0 cache hits
        self.assertEqual(dec1.matched_prefix_tokens, 0)
        chosen_worker = dec1.selected_worker_id

        # Simulate completion of req1
        self.router.release_worker_request(chosen_worker)

        # Second request with identical prefix should hit the same worker with high cache hit ratio
        req2 = RadixRouteRequest("req2", common_prefix + ["user_2"], max_tokens=50)
        dec2 = self.router.route_request(req2)

        self.assertEqual(dec2.selected_worker_id, chosen_worker)
        self.assertGreaterEqual(dec2.matched_prefix_tokens, len(common_prefix))
        self.assertGreater(dec2.cache_hit_ratio, 0.95)


if __name__ == "__main__":
    unittest.main()
