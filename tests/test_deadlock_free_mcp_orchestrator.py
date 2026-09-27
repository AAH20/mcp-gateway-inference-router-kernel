"""
Unit tests for Deadlock-Free Multi-Agent Orchestrator.
"""

import unittest
from mcp_gateway_inference_router_kernel.core.models import AgentTask
from mcp_gateway_inference_router_kernel.core.deadlock_free_mcp_orchestrator import DeadlockFreeMcpOrchestrator


class TestDeadlockFreeMcpOrchestrator(unittest.TestCase):

    def setUp(self):
        self.orchestrator = DeadlockFreeMcpOrchestrator({"lock_a", "lock_b"})

    def test_circular_wait_detection(self):
        # Direct cyclic wait detection: Agent 1 holds lock_a, wants lock_b; Agent 2 holds lock_b, wants lock_a
        held = {"lock_a": "agent_1", "lock_b": "agent_2"}
        pending = {"agent_1": ["lock_b"], "agent_2": ["lock_a"]}
        cycles = self.orchestrator.detect_circular_waits(held, pending)
        self.assertEqual(len(cycles), 1)
        self.assertIn("agent_1", cycles[0])
        self.assertIn("agent_2", cycles[0])

    def test_safe_schedule_execution(self):
        tasks = [
            AgentTask("task_1", "agent_1", "tool_1", ["lock_a"], duration_ms=50.0, priority=2),
            AgentTask("task_2", "agent_2", "tool_2", ["lock_a", "lock_b"], duration_ms=50.0, priority=1),
            AgentTask("task_3", "agent_3", "tool_3", ["lock_b"], duration_ms=40.0, priority=3),
        ]
        report = self.orchestrator.schedule_tasks(tasks, max_concurrency=2)
        self.assertEqual(len(report.safe_sequence), 3)
        self.assertGreater(report.total_makespan_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
