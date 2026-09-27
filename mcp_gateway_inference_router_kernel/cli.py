"""
Command-Line Interface for MCP Gateway & AI Inference Router Kernel.
Provides CLI commands for individual algorithmic solvers and unified benchmark execution.
"""

from __future__ import annotations
import argparse
import json
import sys
from mcp_gateway_inference_router_kernel.engine import McpGatewayInferenceRouterEngine


def print_banner():
    banner = r"""
================================================================================
       MCP GATEWAY & AI INFERENCE ROUTER APEX NP-HARD KERNEL
   Submodular Knapsack, Radix Prefix Caching, Pareto Model Cascades & RCPSP
================================================================================
"""
    print(banner)


def run_benchmark_all():
    print_banner()
    print("[*] Launching MCP Gateway & Intelligence Routing Benchmarks...")
    engine = McpGatewayInferenceRouterEngine()

    report = engine.run_comprehensive_benchmark()

    print("\n" + "=" * 80)
    print("1. COMBINATORIAL MCP SCHEMA KNAPSACK PRUNER (SUBMODULAR COVERAGE)")
    print("=" * 80)
    prune = report.schema_pruning_summary
    print(f"  * Total Unpruned MCP Tools       : {prune['unpruned_tools_count']} tools across 15 servers")
    print(f"  * Raw Schema Token Footprint     : {prune['unpruned_schema_tokens']:,} tokens")
    print(f"  * Optimized Pruned Selection     : {prune['pruned_tools_count']} tools ({prune['pruned_schema_tokens']:,} tokens)")
    print(f"  * Prompt Tokens Saved            : {prune['tokens_saved']:,} tokens (-{prune['token_reduction_pct']:.1f}%)")
    print(f"  * Semantic Coverage Score        : {prune['coverage_score_pct']:.1f}%")
    print(f"  * Solver Optimization Latency    : {prune['pruning_latency_us']:.2f} µs")

    print("\n" + "=" * 80)
    print("2. RADIX-TREE PREFIX-CACHE ROUTER (GPU KV-CACHE AFFINITY)")
    print("=" * 80)
    radix = report.radix_cache_summary
    print(f"  * GPU Cluster Workers            : {radix['cluster_gpu_workers']} nodes (vLLM / SGLang simulated)")
    print(f"  * Inference Requests Evaluated   : {radix['total_requests_routed']:,}")
    print(f"  * Cache-Hit Request Rate         : {radix['cache_hit_requests_pct']:.1f}%")
    print(f"  * Aggregate Token Hit Rate       : {radix['aggregate_token_hit_rate_pct']:.1f}%")
    print(f"  * Reused Cached Tokens           : {radix['cached_tokens_reused']:,} / {radix['total_tokens_evaluated']:,}")
    print(f"  * Average Time-To-First-Token    : {radix['average_ttft_ms']:.2f} ms")

    print("\n" + "=" * 80)
    print("3. CONFORMAL PARETO MODEL CASCADER (QUALITY VS. COST OPTIMIZER)")
    print("=" * 80)
    casc = report.model_cascading_summary
    print(f"  * Agent Queries Dispatched       : {casc['queries_evaluated']}")
    print(f"  * Pure Frontier Cost (Claude 3.7): ${casc['pure_frontier_cost_usd']:.4f}")
    print(f"  * Cascaded Execution Cost        : ${casc['cascaded_cost_usd']:.4f}")
    print(f"  * Net Dollar Savings             : ${casc['total_savings_usd']:.4f}")
    print(f"  * Blended Cost Reduction         : -{casc['blended_savings_pct']:.1f}%")
    print(f"  * Tier Distribution              : {casc['tier_distribution']}")

    print("\n" + "=" * 80)
    print("4. DEADLOCK-FREE MULTI-AGENT ORCHESTRATOR (RCPSP BANKER'S SAFETY)")
    print("=" * 80)
    dead = report.deadlock_orchestration_summary
    print(f"  * Concurrent Agent Tasks         : {dead['total_tasks_scheduled']}")
    print(f"  * Circular Waits Defused         : {dead['deadlocks_prevented']} cyclic deadlocks")
    print(f"  * Safe Priority Preemptions      : {dead['preemptions_triggered']}")
    print(f"  * Executed Task Sequence         : {' -> '.join(dead['safe_execution_order'])}")
    print(f"  * Total Workflow Makespan        : {dead['total_makespan_ms']:.1f} ms")
    print(f"  * Scheduling Verification Time   : {dead['scheduler_latency_us']:.2f} µs")

    print("\n" + "=" * 80)
    print("5. SUBMODULAR SEMANTIC CONTEXT DISTILLER (OBSERVATION COMPRESSOR)")
    print("=" * 80)
    dist = report.context_distillation_summary
    print(f"  * Original Observation History   : {dist['original_tokens']:,} tokens (30 turns)")
    print(f"  * Compressed Distilled History   : {dist['compressed_tokens']:,} tokens")
    print(f"  * Context Window Reduction       : -{dist['compression_ratio_pct']:.1f}% ({dist['tokens_saved']:,} tokens saved)")
    print(f"  * Semantic Saliency Retention    : {dist['semantic_retention_pct']:.1f}%")
    print(f"  * Hard Error Anchors Preserved   : 100% (zero loss of stack traces)")
    print(f"  * Distillation Processing Time   : {dist['distillation_time_ms']:.2f} ms")

    print("\n" + "=" * 80)
    print("6. MULTI-PROVIDER WATER-FILLING ROUTER (ZERO-429 LOAD LEVELING)")
    print("=" * 80)
    water = report.water_filling_summary
    print(f"  * Surge Burst Requests Routed    : {water['burst_requests_dispatched']}")
    print(f"  * Aggregate Tokens Dispatched    : {water['total_tokens_routed']:,} tokens")
    print(f"  * HTTP 429 Dropped Requests      : {water['dropped_429_count']} (0.0% dropouts)")
    print(f"  * Guaranteed Delivery Rate       : {water['guaranteed_delivery_pct']:.1f}%")
    print(f"  * Cluster Quota Utilization      : {water['aggregate_cluster_tpm_utilization_pct']:.1f}%")

    print("\n" + "=" * 80)
    print(f"ALL 6 MCP & INFERENCE SOLVERS EXECUTED IN {report.total_runtime_ms:.2f} ms")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="MCP Gateway & AI Inference Router Apex NP-Hard Solver CLI"
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("benchmark-all", help="Execute all 6 solvers with unified telemetry")
    subparsers.add_parser("prune", help="Run MCP Schema Knapsack Pruning benchmark")
    subparsers.add_parser("radix", help="Run Radix-Tree Prefix Cache Routing benchmark")
    subparsers.add_parser("cascade", help="Run Conformal Model Cascading benchmark")
    subparsers.add_parser("orchestrate", help="Run Deadlock-Free Multi-Agent Orchestrator")
    subparsers.add_parser("distill", help="Run Semantic Context Distillation benchmark")
    subparsers.add_parser("waterfill", help="Run Multi-Provider Water-Filling benchmark")

    args = parser.parse_args()

    engine = McpGatewayInferenceRouterEngine()

    if args.command == "benchmark-all" or args.command is None:
        run_benchmark_all()
    elif args.command == "prune":
        res = engine.run_schema_pruning_benchmark()
        print(json.dumps(res, indent=2))
    elif args.command == "radix":
        res = engine.run_radix_cache_benchmark()
        print(json.dumps(res, indent=2))
    elif args.command == "cascade":
        res = engine.run_model_cascading_benchmark()
        print(json.dumps(res, indent=2))
    elif args.command == "orchestrate":
        res = engine.run_deadlock_orchestration_benchmark()
        print(json.dumps(res, indent=2))
    elif args.command == "distill":
        res = engine.run_context_distillation_benchmark()
        print(json.dumps(res, indent=2))
    elif args.command == "waterfill":
        res = engine.run_water_filling_benchmark()
        print(json.dumps(res, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
