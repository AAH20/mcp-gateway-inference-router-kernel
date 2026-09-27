"""
Unit tests for Semantic Context Distiller.
"""

import unittest
from mcp_gateway_inference_router_kernel.core.models import ObservationChunk
from mcp_gateway_inference_router_kernel.core.semantic_context_distiller import SemanticContextDistiller


class TestSemanticContextDistiller(unittest.TestCase):

    def setUp(self):
        self.distiller = SemanticContextDistiller(target_budget_tokens=1000)

    def test_anchor_preservation_and_compression(self):
        chunks = [
            ObservationChunk("c1", turn_index=1, tool_id="ls", tokens=400, text_content="File listing", saliency_score=0.2),
            ObservationChunk("c2", turn_index=2, tool_id="bash", tokens=300, text_content="Error: NullPointer", saliency_score=0.99, is_hard_anchor=True),
            ObservationChunk("c3", turn_index=3, tool_id="cat", tokens=800, text_content="Long file dump", saliency_score=0.3),
        ]
        res = self.distiller.distill_context(chunks)

        self.assertLessEqual(res.compressed_tokens, 1000)
        self.assertGreater(res.compression_ratio_pct, 0.0)
        # Check that anchor c2 was preserved
        self.assertTrue(any(c.chunk_id == "c2" for c in [c for c in chunks if c.is_hard_anchor]))


if __name__ == "__main__":
    unittest.main()
