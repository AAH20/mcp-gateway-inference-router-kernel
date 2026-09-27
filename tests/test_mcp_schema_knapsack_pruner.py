"""
Unit tests for MCP Schema Knapsack Pruner.
"""

import unittest
from mcp_gateway_inference_router_kernel.core.models import McpTool
from mcp_gateway_inference_router_kernel.core.mcp_schema_knapsack_pruner import McpSchemaKnapsackPruner


class TestMcpSchemaKnapsackPruner(unittest.TestCase):

    def setUp(self):
        # 5 tools: Tool B depends on Tool A
        self.tools = [
            McpTool("tool_a", "postgres_connect", "postgres", schema_tokens=400, prerequisites=[], semantic_tags=["postgres", "db"]),
            McpTool("tool_b", "postgres_query", "postgres", schema_tokens=500, prerequisites=["tool_a"], semantic_tags=["postgres", "query"]),
            McpTool("tool_c", "github_commit", "github", schema_tokens=600, prerequisites=[], semantic_tags=["github", "git"]),
            McpTool("tool_d", "slack_post", "slack", schema_tokens=450, prerequisites=[], semantic_tags=["slack", "chat"]),
            McpTool("tool_e", "bash_exec", "bash", schema_tokens=700, prerequisites=[], semantic_tags=["bash", "shell"]),
        ]
        self.pruner = McpSchemaKnapsackPruner(self.tools)

    def test_pruning_respects_token_budget(self):
        res = self.pruner.prune_tools(query_keywords={"postgres", "query"}, token_budget=1000)
        self.assertLessEqual(res.total_tokens, 1000)
        self.assertGreater(len(res.selected_tools), 0)

    def test_prerequisite_closure_enforcement(self):
        # If tool_b is selected, tool_a MUST be included in the selection
        res = self.pruner.prune_tools(query_keywords={"postgres", "query"}, token_budget=2000)
        selected_ids = {t.tool_id for t in res.selected_tools}
        if "tool_b" in selected_ids:
            self.assertIn("tool_a", selected_ids)


if __name__ == "__main__":
    unittest.main()
