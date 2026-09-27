"""
Integrated Execution Engine for MCP Gateway & AI Inference Router Kernel.
Runs sub-millisecond benchmarks across all 6 core algorithmic modules and aggregates metrics.
"""

from __future__ import annotations
import datetime
import random
import time
from typing import Dict, Any, List
from mcp_gateway_inference_router_kernel.core.models import (
    McpTool,
    InferenceWorker,
    RadixRouteRequest,
    QueryCharacteristics,
    AgentTask,
    ObservationChunk,
    ProviderEndpoint,
    McpRouterBenchmarkReport,
)
from mcp_gateway_inference_router_kernel.core.mcp_schema_knapsack_pruner import McpSchemaKnapsackPruner
from mcp_gateway_inference_router_kernel.core.radix_prefix_cache_router import RadixPrefixCacheRouter
from mcp_gateway_inference_router_kernel.core.conformal_model_cascader import ConformalModelCascader
from mcp_gateway_inference_router_kernel.core.deadlock_free_mcp_orchestrator import DeadlockFreeMcpOrchestrator
from mcp_gateway_inference_router_kernel.core.semantic_context_distiller import SemanticContextDistiller
from mcp_gateway_inference_router_kernel.core.multi_provider_water_filling import MultiProviderWaterFillingRouter


class McpGatewayInferenceRouterEngine:
    """Orchestrates comprehensive algorithmic benchmarks for MCP gateways and inference routers."""

    def __init__(self):
        pass

    def run_schema_pruning_benchmark(self, num_tools: int = 200) -> Dict[str, Any]:
        """Benchmarks submodular knapsack pruning on 200 tools across 15 MCP servers."""
        prng = random.Random(42)
        servers = ["github", "postgres", "slack", "bash", "browser", "aws", "jira", "docker", "s3", "linear", "notion", "gmail", "datadog", "kubernetes", "stripe"]

        tools: List[McpTool] = []
        for i in range(num_tools):
            srv = servers[i % len(servers)]
            t_id = f"tool_{srv}_{i}"
            # Some tools have prerequisites
            prereqs = [f"tool_{srv}_{i-1}"] if (i % 5 == 0 and i > 0) else []
            tools.append(McpTool(
                tool_id=t_id,
                name=f"{srv}_{['create', 'query', 'delete', 'list', 'update'][i % 5]}_{i}",
                server_name=srv,
                schema_tokens=prng.randint(300, 750),
                prerequisites=prereqs,
                semantic_tags=[srv, "data", "api", "query", "code"][i % 5 : (i % 5) + 3],
                utility_weight=prng.uniform(0.5, 2.0),
            ))

        pruner = McpSchemaKnapsackPruner(tools)
        # Search for tools relevant to database query and github commit
        query_keys = {"postgres", "github", "query", "code"}
        res = pruner.prune_tools(query_keywords=query_keys, token_budget=4000)

        return {
            "unpruned_tools_count": num_tools,
            "unpruned_schema_tokens": res.unpruned_tokens,
            "pruned_tools_count": len(res.selected_tools),
            "pruned_schema_tokens": res.total_tokens,
            "tokens_saved": res.tokens_saved,
            "token_reduction_pct": res.prune_ratio_pct,
            "coverage_score_pct": res.coverage_score * 100.0,
            "pruning_latency_us": res.latency_us,
        }

    def run_radix_cache_benchmark(self, num_requests: int = 300) -> Dict[str, Any]:
        """Benchmarks Radix-Tree prefix cache routing on an 8-worker inference cluster."""
        workers = [
            InferenceWorker(f"gpu_worker_{i}", f"10.0.0.{10+i}", kv_capacity_tokens=64000, active_requests=0)
            for i in range(8)
        ]
        router = RadixPrefixCacheRouter(workers)

        shared_system_prefix = [f"sys_tok_{j}" for j in range(200)]
        mcp_tool_prefix = [f"mcp_schema_tok_{j}" for j in range(400)]
        base_prefix = shared_system_prefix + mcp_tool_prefix

        # Route requests with high shared prefix locality (simulating multi-agent conversations)
        hits = 0
        total_tokens = 0
        hit_tokens = 0
        ttft_sum = 0.0

        for i in range(num_requests):
            user_suffix = [f"user_{i % 10}_tok_{k}" for k in range(50 + (i % 20))]
            full_prompt = base_prefix + user_suffix

            req = RadixRouteRequest(
                request_id=f"req_{i}",
                prompt_tokens=full_prompt,
                max_tokens=100,
            )
            decision = router.route_request(req)

            total_tokens += len(full_prompt)
            hit_tokens += decision.matched_prefix_tokens
            ttft_sum += decision.estimated_ttft_ms
            if decision.matched_prefix_tokens > 0:
                hits += 1

            # Periodically release completed requests
            if i % 3 == 0:
                router.release_worker_request(decision.selected_worker_id)

        agg_hit_rate = (hit_tokens / total_tokens * 100.0) if total_tokens > 0 else 0.0
        avg_ttft = ttft_sum / num_requests if num_requests > 0 else 0.0

        return {
            "cluster_gpu_workers": len(workers),
            "total_requests_routed": num_requests,
            "cache_hit_requests_pct": (hits / num_requests * 100.0),
            "aggregate_token_hit_rate_pct": agg_hit_rate,
            "average_ttft_ms": avg_ttft,
            "total_tokens_evaluated": total_tokens,
            "cached_tokens_reused": hit_tokens,
        }

    def run_model_cascading_benchmark(self, num_queries: int = 150) -> Dict[str, Any]:
        """Benchmarks multi-objective Pareto model cascading."""
        prng = random.Random(123)
        cascader = ConformalModelCascader()

        decisions = []
        total_cost_cascaded = 0.0
        total_cost_pure_frontier = 0.0

        for i in range(num_queries):
            q = QueryCharacteristics(
                query_id=f"q_{i}",
                token_length=prng.randint(800, 6000),
                code_density=prng.uniform(0.1, 0.9),
                reasoning_depth_est=prng.uniform(0.1, 0.85),
                required_tool_count=prng.randint(1, 8),
                max_budget_usd=0.05,
                latency_deadline_ms=2000.0,
            )
            dec = cascader.route_query(q)
            decisions.append(dec)

            # Pure frontier baseline: Claude 3.7 Sonnet ($3.00/1M input, $15.00/1M output)
            frontier_cost = (q.token_length / 1e6 * 3.00) + (500 / 1e6 * 15.00)
            total_cost_pure_frontier += frontier_cost
            total_cost_cascaded += dec.actual_cost_usd

        savings_usd = total_cost_pure_frontier - total_cost_cascaded
        savings_pct = (savings_usd / total_cost_pure_frontier * 100.0) if total_cost_pure_frontier > 0 else 0.0

        tier_counts = {
            "CHEAP_FAST": sum(1 for d in decisions if d.routed_tier.value == "CHEAP_FAST"),
            "BALANCED": sum(1 for d in decisions if d.routed_tier.value == "BALANCED"),
            "FRONTIER_REASONER": sum(1 for d in decisions if d.routed_tier.value == "FRONTIER_REASONER"),
        }

        return {
            "queries_evaluated": num_queries,
            "pure_frontier_cost_usd": total_cost_pure_frontier,
            "cascaded_cost_usd": total_cost_cascaded,
            "total_savings_usd": savings_usd,
            "blended_savings_pct": savings_pct,
            "tier_distribution": tier_counts,
        }

    def run_deadlock_orchestration_benchmark(self) -> Dict[str, Any]:
        """Benchmarks deadlock-free multi-agent lock scheduling."""
        resources = {"git_repo_lock", "postgres_txn_lock", "docker_container_lock", "browser_session_lock"}
        orchestrator = DeadlockFreeMcpOrchestrator(resources)

        # Create tasks with intentional cyclic lock dependencies (Agent A wants [git, db], Agent B wants [db, git])
        tasks = [
            AgentTask("task_1", "agent_coder", "git_commit", ["git_repo_lock", "postgres_txn_lock"], duration_ms=45.0, priority=3),
            AgentTask("task_2", "agent_db_migrator", "db_migrate", ["postgres_txn_lock", "git_repo_lock"], duration_ms=60.0, priority=2),
            AgentTask("task_3", "agent_qa", "browser_test", ["browser_session_lock", "docker_container_lock"], duration_ms=80.0, priority=1),
            AgentTask("task_4", "agent_deployer", "docker_build", ["docker_container_lock", "git_repo_lock"], duration_ms=50.0, priority=2),
        ]

        report = orchestrator.schedule_tasks(tasks)

        return {
            "total_tasks_scheduled": report.total_tasks,
            "deadlocks_prevented": report.circular_waits_prevented,
            "preemptions_triggered": report.preemptions_triggered,
            "safe_execution_order": report.safe_sequence,
            "total_makespan_ms": report.total_makespan_ms,
            "scheduler_latency_us": report.execution_time_us,
        }

    def run_context_distillation_benchmark(self) -> Dict[str, Any]:
        """Benchmarks submodular semantic context distillation on 30-turn agent observations."""
        chunks: List[ObservationChunk] = []
        for i in range(30):
            is_anchor = (i == 28)  # Critical traceback on turn 28
            tokens = 800 if not is_anchor else 450
            chunks.append(ObservationChunk(
                chunk_id=f"obs_chunk_{i}",
                turn_index=i + 1,
                tool_id="bash_exec" if i % 2 == 0 else "postgres_query",
                tokens=tokens,
                text_content=f"Observation logs turn {i+1} [Traceback: RuntimeError at L42]" if is_anchor else f"Standard row stdout {i} " * 60,
                saliency_score=0.98 if is_anchor else 0.35 + (i * 0.015),
                is_hard_anchor=is_anchor,
            ))

        distiller = SemanticContextDistiller(target_budget_tokens=4000)
        res = distiller.distill_context(chunks)

        return {
            "original_tokens": res.original_tokens,
            "compressed_tokens": res.compressed_tokens,
            "tokens_saved": res.original_tokens - res.compressed_tokens,
            "compression_ratio_pct": res.compression_ratio_pct,
            "semantic_retention_pct": res.semantic_retention_score * 100.0,
            "retained_chunks": res.retained_chunks_count,
            "distillation_time_ms": res.distillation_time_ms,
        }

    def run_water_filling_benchmark(self, num_burst_requests: int = 100) -> Dict[str, Any]:
        """Benchmarks multi-provider token-bucket water-filling load leveling."""
        providers = [
            ProviderEndpoint("anthropic", "Anthropic Claude API", rpm_limit=1000, tpm_limit=80000, current_rpm=0, current_tpm=0, error_rate_429=0.0, cost_per_1k_tokens=0.003),
            ProviderEndpoint("openai", "OpenAI GPT-4o API", rpm_limit=1200, tpm_limit=90000, current_rpm=0, current_tpm=0, error_rate_429=0.0, cost_per_1k_tokens=0.0025),
            ProviderEndpoint("bedrock", "AWS Bedrock Claude", rpm_limit=800, tpm_limit=60000, current_rpm=0, current_tpm=0, error_rate_429=0.0, cost_per_1k_tokens=0.003),
            ProviderEndpoint("groq", "Groq Llama-3 API", rpm_limit=2000, tpm_limit=120000, current_rpm=0, current_tpm=0, error_rate_429=0.0, cost_per_1k_tokens=0.0005),
        ]

        router = MultiProviderWaterFillingRouter(providers)

        dropped_429 = 0
        allocated_count = 0
        total_tokens_routed = 0

        for i in range(num_burst_requests):
            req_tokens = 1500 + (i % 10) * 200
            alloc = router.allocate_request(req_tokens, is_interactive=(i % 2 == 0))
            if not alloc.guaranteed_no_429:
                dropped_429 += 1
            allocated_count += 1
            total_tokens_routed += req_tokens

        total_tpm_cap = sum(p.tpm_limit for p in providers)
        current_tpm_used = sum(p.current_tpm for p in providers)
        util_pct = (current_tpm_used / total_tpm_cap * 100.0)

        return {
            "burst_requests_dispatched": num_burst_requests,
            "total_tokens_routed": total_tokens_routed,
            "dropped_429_count": dropped_429,
            "guaranteed_delivery_pct": ((allocated_count - dropped_429) / allocated_count * 100.0),
            "aggregate_cluster_tpm_utilization_pct": util_pct,
            "active_provider_endpoints": len(providers),
        }

    def run_comprehensive_benchmark(self) -> McpRouterBenchmarkReport:
        """Executes all 6 solvers into unified benchmark telemetry."""
        t0 = time.perf_counter()

        prune_res = self.run_schema_pruning_benchmark()
        radix_res = self.run_radix_cache_benchmark()
        cascade_res = self.run_model_cascading_benchmark()
        deadlock_res = self.run_deadlock_orchestration_benchmark()
        distill_res = self.run_context_distillation_benchmark()
        water_res = self.run_water_filling_benchmark()

        total_runtime_ms = (time.perf_counter() - t0) * 1000.0

        return McpRouterBenchmarkReport(
            total_runtime_ms=total_runtime_ms,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            schema_pruning_summary=prune_res,
            radix_cache_summary=radix_res,
            model_cascading_summary=cascade_res,
            deadlock_orchestration_summary=deadlock_res,
            context_distillation_summary=distill_res,
            water_filling_summary=water_res,
        )
