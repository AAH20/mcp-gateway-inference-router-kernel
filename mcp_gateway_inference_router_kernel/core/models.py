"""
Data models and typed structures for MCP Gateway & AI Inference Router Kernel.
Zero external pip dependencies. Strict Python 3.10+ typing.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set


class ModelTierClass(str, Enum):
    CHEAP_FAST = "CHEAP_FAST"             # e.g., Gemini 2.5 Flash, DeepSeek V3 ($0.075 - $0.20 / 1M)
    BALANCED = "BALANCED"                 # e.g., Claude 3.5 Haiku, GPT-4o-mini ($0.80 - $1.00 / 1M)
    FRONTIER_REASONER = "FRONTIER_REASONER" # e.g., Claude 3.7 Sonnet, DeepSeek R1, GPT-4.5 ($3.00 - $30.00 / 1M)


@dataclass(slots=True)
class McpTool:
    tool_id: str
    name: str
    server_name: str
    schema_tokens: int
    prerequisites: List[str] = field(default_factory=list)
    semantic_tags: List[str] = field(default_factory=list)
    utility_weight: float = 1.0


@dataclass(slots=True)
class PrunedToolSelection:
    selected_tools: List[McpTool]
    total_tokens: int
    unpruned_tokens: int
    tokens_saved: int
    prune_ratio_pct: float
    coverage_score: float
    latency_us: float


@dataclass(slots=True)
class InferenceWorker:
    worker_id: str
    hostname: str
    kv_capacity_tokens: int
    active_requests: int
    latency_penalty_factor: float = 1.0


@dataclass(slots=True)
class RadixRouteRequest:
    request_id: str
    prompt_tokens: List[str]  # Simulated token strings or hashes
    max_tokens: int
    priority: int = 1


@dataclass(slots=True)
class RadixRouteDecision:
    request_id: str
    selected_worker_id: str
    matched_prefix_tokens: int
    cache_hit_ratio: float
    estimated_prefill_tokens: int
    estimated_ttft_ms: float
    routing_latency_us: float


@dataclass(slots=True)
class ModelTier:
    tier: ModelTierClass
    name: str
    cost_per_1m_input: float
    cost_per_1m_output: float
    base_latency_ms: float
    quality_score: float  # Benchmark tool-calling accuracy [0, 1]


@dataclass(slots=True)
class QueryCharacteristics:
    query_id: str
    token_length: int
    code_density: float           # [0, 1]
    reasoning_depth_est: float    # [0, 1]
    required_tool_count: int
    max_budget_usd: float
    latency_deadline_ms: float


@dataclass(slots=True)
class CascadeDecision:
    query_id: str
    routed_tier: ModelTierClass
    was_speculative: bool
    speculative_accepted: bool
    actual_cost_usd: float
    latency_ms: float
    quality_confidence: float
    cost_savings_pct: float


@dataclass(slots=True)
class AgentTask:
    task_id: str
    agent_id: str
    tool_id: str
    required_locks: List[str]
    duration_ms: float
    priority: int = 1


@dataclass(slots=True)
class DeadlockPreventionReport:
    total_tasks: int
    safe_sequence: List[str]
    preemptions_triggered: int
    circular_waits_prevented: int
    total_makespan_ms: float
    execution_time_us: float


@dataclass(slots=True)
class ObservationChunk:
    chunk_id: str
    turn_index: int
    tool_id: str
    tokens: int
    text_content: str
    saliency_score: float
    is_hard_anchor: bool = False  # e.g., error trace, key auth token, output schema


@dataclass(slots=True)
class DistilledContext:
    original_tokens: int
    compressed_tokens: int
    compression_ratio_pct: float
    semantic_retention_score: float
    retained_chunks_count: int
    distillation_time_ms: float


@dataclass(slots=True)
class ProviderEndpoint:
    provider_id: str
    name: str
    rpm_limit: int
    tpm_limit: int
    current_rpm: int
    current_tpm: int
    error_rate_429: float
    cost_per_1k_tokens: float


@dataclass(slots=True)
class WaterFillingAllocation:
    allocated_provider_id: str
    request_tokens: int
    queue_delay_ms: float
    guaranteed_no_429: bool
    aggregate_tpm_utilization: float


@dataclass(slots=True)
class McpRouterBenchmarkReport:
    total_runtime_ms: float
    timestamp: str
    schema_pruning_summary: Dict[str, float]
    radix_cache_summary: Dict[str, float]
    model_cascading_summary: Dict[str, float]
    deadlock_orchestration_summary: Dict[str, float]
    context_distillation_summary: Dict[str, float]
    water_filling_summary: Dict[str, float]
