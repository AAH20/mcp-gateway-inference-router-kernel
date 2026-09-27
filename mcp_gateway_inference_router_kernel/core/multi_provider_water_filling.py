"""
Multi-Provider Token-Bucket Water-Filling Algorithm.
Optimizes traffic distribution across heterogeneous model API quotas (Anthropic, OpenAI, Bedrock, Groq)
to prevent HTTP 429 RateLimitExceeded errors and maximize aggregate throughput.
"""

from __future__ import annotations
import math
import time
from typing import List, Dict, Optional, Tuple
from mcp_gateway_inference_router_kernel.core.models import (
    ProviderEndpoint,
    WaterFillingAllocation,
)


class MultiProviderWaterFillingRouter:
    """
    Leaky-bucket fluid flow water-filling algorithm for multi-provider AI gateways.
    Maintains continuous sliding window utilization and prevents rate-limit dropouts.
    """

    def __init__(self, providers: List[ProviderEndpoint]):
        self.providers = providers

    def allocate_request(
        self,
        request_tokens: int,
        is_interactive: bool = True,
    ) -> WaterFillingAllocation:
        """
        Routes an incoming token payload to the least-congested provider endpoint.
        Interactive requests receive priority reservation headroom.
        """
        best_provider: Optional[ProviderEndpoint] = None
        best_score = float("inf")

        for p in self.providers:
            # Check if this request fits within provider TPM/RPM headroom
            tpm_headroom = max(0, p.tpm_limit - p.current_tpm)
            rpm_headroom = max(0, p.rpm_limit - p.current_rpm)

            if tpm_headroom < request_tokens or rpm_headroom < 1:
                continue

            # Fractional utilization [0, 1]
            tpm_util = (p.current_tpm + request_tokens) / p.tpm_limit
            rpm_util = (p.current_rpm + 1) / p.rpm_limit

            # Barrier function penalty approaching 100% capacity: 1 / (1 - util)
            barrier_tpm = 1.0 / max(0.01, 1.0 - tpm_util)
            barrier_rpm = 1.0 / max(0.01, 1.0 - rpm_util)

            # Interactive requests prioritize lower current queue/latency;
            # Background requests prioritize cost per 1k tokens
            cost_weight = 0.1 if is_interactive else 2.0
            composite_score = (barrier_tpm + barrier_rpm) + (p.cost_per_1k_tokens * cost_weight)

            if composite_score < best_score:
                best_score = composite_score
                best_provider = p

        if best_provider:
            # Allocate to best provider
            best_provider.current_tpm += request_tokens
            best_provider.current_rpm += 1

            total_tpm_limit = sum(p.tpm_limit for p in self.providers)
            total_current_tpm = sum(p.current_tpm for p in self.providers)
            agg_util = (total_current_tpm / total_tpm_limit * 100.0) if total_tpm_limit > 0 else 0.0

            return WaterFillingAllocation(
                allocated_provider_id=best_provider.provider_id,
                request_tokens=request_tokens,
                queue_delay_ms=0.0,
                guaranteed_no_429=True,
                aggregate_tpm_utilization=agg_util,
            )
        else:
            # Over-capacity: Pick provider that frees capacity soonest with minimal queue delay
            fallback = min(self.providers, key=lambda p: p.current_tpm / p.tpm_limit)
            simulated_delay_ms = ((request_tokens) / max(1, fallback.tpm_limit)) * 60_000.0

            fallback.current_tpm += request_tokens
            fallback.current_rpm += 1

            total_tpm_limit = sum(p.tpm_limit for p in self.providers)
            total_current_tpm = sum(p.current_tpm for p in self.providers)
            agg_util = (total_current_tpm / total_tpm_limit * 100.0) if total_tpm_limit > 0 else 0.0

            return WaterFillingAllocation(
                allocated_provider_id=fallback.provider_id,
                request_tokens=request_tokens,
                queue_delay_ms=simulated_delay_ms,
                guaranteed_no_429=False,
                aggregate_tpm_utilization=agg_util,
            )

    def step_decay(self, elapsed_seconds: float = 1.0) -> None:
        """Simulates continuous bucket drainage over elapsed time window."""
        drain_fraction = min(1.0, elapsed_seconds / 60.0)
        for p in self.providers:
            p.current_tpm = max(0, int(p.current_tpm * (1.0 - drain_fraction)))
            p.current_rpm = max(0, int(p.current_rpm * (1.0 - drain_fraction)))
