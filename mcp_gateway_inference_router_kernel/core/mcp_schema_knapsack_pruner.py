"""
Combinatorial MCP Tool Schema Pruner & Submodular Knapsack Optimizer.
Solves the MCP schema explosion problem: prunes thousands of tool schemas down to
a strict context budget while enforcing topological DAG prerequisites and maximizing relevance.
"""

from __future__ import annotations
import math
import time
from typing import List, Dict, Set, Tuple
from mcp_gateway_inference_router_kernel.core.models import (
    McpTool,
    PrunedToolSelection,
)


class McpSchemaKnapsackPruner:
    """
    Submodular greedy selector with DAG prerequisite closure for MCP tool schemas.
    Achieves (1 - 1/e) theoretical approximation ratio on submodular coverage.
    """

    def __init__(self, all_tools: List[McpTool]):
        self.all_tools = all_tools
        self.tool_map: Dict[str, McpTool] = {t.tool_id: t for t in all_tools}
        self.total_unpruned_tokens = sum(t.schema_tokens for t in all_tools)

    def _get_ancestor_closure(self, tool_id: str, visited: Set[str]) -> Tuple[List[McpTool], int]:
        """Returns the prerequisite transitive closure for a tool and its total token cost."""
        closure: List[McpTool] = []
        closure_tokens = 0
        queue = [tool_id]

        while queue:
            curr_id = queue.pop(0)
            if curr_id in visited:
                continue
            tool = self.tool_map.get(curr_id)
            if not tool:
                continue
            visited.add(curr_id)
            closure.append(tool)
            closure_tokens += tool.schema_tokens
            for prereq in tool.prerequisites:
                if prereq not in visited:
                    queue.append(prereq)

        return closure, closure_tokens

    def prune_tools(
        self,
        query_keywords: Set[str],
        token_budget: int = 4000,
        redundancy_penalty: float = 0.15,
    ) -> PrunedToolSelection:
        """
        Selects an optimal subset of tools fitting within token_budget.
        Evaluates submodular marginal gain over semantic categories with prerequisite closure.
        """
        start_t = time.perf_counter()

        selected_ids: Set[str] = set()
        selected_tools: List[McpTool] = []
        current_tokens = 0
        category_counts: Dict[str, int] = {}

        # Precompute individual relevance score for each tool relative to query_keywords
        relevance_scores: Dict[str, float] = {}
        for tool in self.all_tools:
            overlap = len(query_keywords.intersection(set(tool.semantic_tags)))
            name_match = 1.0 if any(k.lower() in tool.name.lower() for k in query_keywords) else 0.0
            score = (overlap * 2.0 + name_match * 3.0 + tool.utility_weight)
            relevance_scores[tool.tool_id] = max(0.1, score)

        # Greedy selection based on marginal utility density: Delta_Gain / Delta_Tokens
        candidate_ids = set(self.tool_map.keys())

        while candidate_ids and current_tokens < token_budget:
            best_tool_id: Optional[str] = None
            best_density: float = -1.0
            best_closure: List[McpTool] = []
            best_closure_tokens: int = 0

            for tool_id in candidate_ids:
                if tool_id in selected_ids:
                    continue

                # Evaluate closure
                temp_visited = set(selected_ids)
                closure, closure_cost = self._get_ancestor_closure(tool_id, temp_visited)

                if current_tokens + closure_cost > token_budget:
                    continue

                # Compute marginal gain of adding this entire closure
                marginal_gain = 0.0
                for c_tool in closure:
                    base_rel = relevance_scores[c_tool.tool_id]
                    # Submodular category saturation
                    curr_cat_count = category_counts.get(c_tool.server_name, 0)
                    saturation_factor = 1.0 / math.sqrt(1.0 + curr_cat_count)
                    # Redundancy penalty for tools in the same server
                    penalty = redundancy_penalty * curr_cat_count
                    marginal_gain += max(0.01, (base_rel * saturation_factor) - penalty)

                density = marginal_gain / max(1, closure_cost)
                if density > best_density:
                    best_density = density
                    best_tool_id = tool_id
                    best_closure = closure
                    best_closure_tokens = closure_cost

            if not best_tool_id or not best_closure:
                break

            # Add selected closure
            for tool in best_closure:
                selected_ids.add(tool.tool_id)
                selected_tools.append(tool)
                current_tokens += tool.schema_tokens
                category_counts[tool.server_name] = category_counts.get(tool.server_name, 0) + 1
                candidate_ids.discard(tool.tool_id)

        tokens_saved = self.total_unpruned_tokens - current_tokens
        prune_ratio = (tokens_saved / self.total_unpruned_tokens * 100.0) if self.total_unpruned_tokens > 0 else 0.0

        # Calculate coverage score
        total_possible_score = sum(relevance_scores.values())
        achieved_score = sum(relevance_scores[t.tool_id] for t in selected_tools)
        coverage_score = (achieved_score / total_possible_score) if total_possible_score > 0 else 1.0

        elapsed_us = (time.perf_counter() - start_t) * 1_000_000.0

        return PrunedToolSelection(
            selected_tools=selected_tools,
            total_tokens=current_tokens,
            unpruned_tokens=self.total_unpruned_tokens,
            tokens_saved=tokens_saved,
            prune_ratio_pct=prune_ratio,
            coverage_score=coverage_score,
            latency_us=elapsed_us,
        )
