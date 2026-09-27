"""
Radix-Tree Prefix-Cache Aware Distributed Inference Router.
Solves the Capacitated Metric Facility Location problem on distributed Trie caches:
dispatches prompt tokens to the optimal GPU inference worker maximizing KV-cache reuse.
"""

from __future__ import annotations
import time
from typing import List, Dict, Optional, Tuple
from mcp_gateway_inference_router_kernel.core.models import (
    InferenceWorker,
    RadixRouteRequest,
    RadixRouteDecision,
)


class RadixTrieNode:
    """Radix tree node representing a sequence of cached prompt tokens in GPU HBM."""
    __slots__ = ("token", "children", "access_count", "last_accessed_ns")

    def __init__(self, token: str):
        self.token = token
        self.children: Dict[str, RadixTrieNode] = {}
        self.access_count = 1
        self.last_accessed_ns = time.perf_counter_ns()


class WorkerRadixCache:
    """Maintains the simulated GPU KV-cache Radix Tree for a single inference worker."""

    def __init__(self, worker: InferenceWorker):
        self.worker = worker
        self.root = RadixTrieNode("<ROOT>")
        self.total_cached_tokens = 0

    def find_lcp(self, tokens: List[str]) -> int:
        """Finds the Longest Common Prefix (LCP) length matching cached tokens."""
        curr = self.root
        matched = 0
        for tok in tokens:
            if tok in curr.children:
                curr = curr.children[tok]
                matched += 1
            else:
                break
        return matched

    def insert_tokens(self, tokens: List[str]) -> int:
        """Inserts token sequence into radix tree, updating cache size."""
        curr = self.root
        newly_inserted = 0
        now = time.perf_counter_ns()

        for tok in tokens:
            if tok not in curr.children:
                curr.children[tok] = RadixTrieNode(tok)
                newly_inserted += 1
                self.total_cached_tokens += 1
            curr = curr.children[tok]
            curr.access_count += 1
            curr.last_accessed_ns = now

        return newly_inserted


class RadixPrefixCacheRouter:
    """
    Distributes inference queries across a cluster of vLLM / SGLang workers.
    Balances Longest Common Prefix (LCP) cache reuse against worker queue delays.
    """

    def __init__(
        self,
        workers: List[InferenceWorker],
        prefill_cost_per_token_ms: float = 0.05,
        queue_penalty_ms_per_request: float = 12.0,
    ):
        self.workers = workers
        self.worker_caches: Dict[str, WorkerRadixCache] = {
            w.worker_id: WorkerRadixCache(w) for w in workers
        }
        self.prefill_cost = prefill_cost_per_token_ms
        self.queue_penalty = queue_penalty_ms_per_request

        # Telemetry
        self.total_routed_requests = 0
        self.total_prompt_tokens = 0
        self.total_cache_hit_tokens = 0

    def route_request(self, req: RadixRouteRequest) -> RadixRouteDecision:
        """
        Evaluates effective TTFT across all worker nodes and selects the minimum latency target.
        Cost = (PromptLen - LCP) * prefill_cost + QueueDelay
        """
        start_t = time.perf_counter()
        self.total_routed_requests += 1

        prompt_len = len(req.prompt_tokens)
        self.total_prompt_tokens += prompt_len

        best_worker_id: Optional[str] = None
        best_cost_ms = float("inf")
        best_matched_lcp = 0

        for w in self.workers:
            cache = self.worker_caches[w.worker_id]
            lcp = cache.find_lcp(req.prompt_tokens)
            uncached_tokens = max(0, prompt_len - lcp)

            prefill_time = uncached_tokens * self.prefill_cost * w.latency_penalty_factor
            queue_delay = w.active_requests * self.queue_penalty

            effective_cost_ms = prefill_time + queue_delay

            if effective_cost_ms < best_cost_ms:
                best_cost_ms = effective_cost_ms
                best_worker_id = w.worker_id
                best_matched_lcp = lcp

        if not best_worker_id:
            best_worker_id = self.workers[0].worker_id

        # Update selected worker's cache and active count
        chosen_cache = self.worker_caches[best_worker_id]
        chosen_cache.insert_tokens(req.prompt_tokens)
        chosen_worker = next(w for w in self.workers if w.worker_id == best_worker_id)
        chosen_worker.active_requests += 1

        self.total_cache_hit_tokens += best_matched_lcp
        hit_ratio = (best_matched_lcp / prompt_len) if prompt_len > 0 else 0.0
        est_prefill = prompt_len - best_matched_lcp

        elapsed_us = (time.perf_counter() - start_t) * 1_000_000.0

        return RadixRouteDecision(
            request_id=req.request_id,
            selected_worker_id=best_worker_id,
            matched_prefix_tokens=best_matched_lcp,
            cache_hit_ratio=hit_ratio,
            estimated_prefill_tokens=est_prefill,
            estimated_ttft_ms=best_cost_ms,
            routing_latency_us=elapsed_us,
        )

    def release_worker_request(self, worker_id: str) -> None:
        """Simulates completion of an active request on a worker node."""
        for w in self.workers:
            if w.worker_id == worker_id:
                w.active_requests = max(0, w.active_requests - 1)
                break
