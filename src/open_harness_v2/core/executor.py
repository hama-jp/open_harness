"""Executor — runs tool calls with policy checks.

Supports both sequential and concurrent tool execution.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from open_harness_v2.events.bus import EventBus
from open_harness_v2.policy.engine import PolicyEngine, PolicyViolation
from open_harness_v2.tools.registry import ToolRegistry
from open_harness_v2.types import AgentEvent, EventType, ToolCall, ToolResult

_logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing one or more tool calls."""

    results: list[tuple[ToolCall, ToolResult]] = field(default_factory=list)
    violations: list[tuple[ToolCall, PolicyViolation]] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        return (
            not self.violations
            and all(r.success for _, r in self.results)
        )


class Executor:
    """Runs tool calls through policy checks and the tool registry.

    Usage::

        executor = Executor(registry, policy, event_bus)
        result = await executor.execute(tool_calls)
    """

    def __init__(
        self,
        registry: ToolRegistry,
        policy: PolicyEngine | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._event_bus = event_bus

    async def execute(
        self,
        tool_calls: list[ToolCall],
        concurrent: bool = False,
    ) -> ExecutionResult:
        """Execute tool calls sequentially (default) or concurrently.

        Parameters
        ----------
        tool_calls:
            List of tool calls to execute.
        concurrent:
            If True, run all tool calls concurrently via asyncio.gather.
        """
        if concurrent and len(tool_calls) > 1:
            return await self._execute_concurrent(tool_calls)
        return await self._execute_sequential(tool_calls)

    async def _execute_sequential(
        self, tool_calls: list[ToolCall]
    ) -> ExecutionResult:
        exec_result = ExecutionResult()
        for tc in tool_calls:
            # Policy check
            if self._policy:
                violation = self._policy.check(tc.name, tc.arguments)
                if violation:
                    exec_result.violations.append((tc, violation))
                    await self._emit(EventType.POLICY_VIOLATION, {
                        "tool": tc.name,
                        "rule": violation.rule,
                        "message": violation.message,
                    })
                    # Return violation as a failed ToolResult so the agent can adapt
                    exec_result.results.append((
                        tc,
                        ToolResult(
                            success=False,
                            output="",
                            error=f"Policy violation: {violation.message}",
                        ),
                    ))
                    continue

            # Execute
            await self._emit(EventType.TOOL_EXECUTING, {
                "tool": tc.name,
                "arguments": tc.arguments,
            })

            result = await self._registry.execute(tc.name, tc.arguments)

            # Record in policy budget
            if self._policy:
                self._policy.record(tc.name)

            if result.success:
                await self._emit(EventType.TOOL_EXECUTED, {
                    "tool": tc.name,
                    "output_length": len(result.output),
                })
            else:
                await self._emit(EventType.TOOL_ERROR, {
                    "tool": tc.name,
                    "error": result.error,
                })

            exec_result.results.append((tc, result))

        return exec_result

    async def _execute_concurrent(
        self, tool_calls: list[ToolCall]
    ) -> ExecutionResult:
        """Execute all tool calls concurrently."""
        exec_result = ExecutionResult()

        # First, check all policies
        to_run: list[ToolCall] = []
        for tc in tool_calls:
            if self._policy:
                violation = self._policy.check(tc.name, tc.arguments)
                if violation:
                    exec_result.violations.append((tc, violation))
                    exec_result.results.append((
                        tc,
                        ToolResult(
                            success=False,
                            output="",
                            error=f"Policy violation: {violation.message}",
                        ),
                    ))
                    continue
            to_run.append(tc)

        if not to_run:
            return exec_result

        # Execute concurrently
        async def _run_one(tc: ToolCall) -> tuple[ToolCall, ToolResult]:
            await self._emit(EventType.TOOL_EXECUTING, {
                "tool": tc.name,
                "arguments": tc.arguments,
            })
            result = await self._registry.execute(tc.name, tc.arguments)
            if self._policy:
                self._policy.record(tc.name)
            return tc, result

        pairs = await asyncio.gather(
            *[_run_one(tc) for tc in to_run],
            return_exceptions=True,
        )

        for item in pairs:
            if isinstance(item, Exception):
                _logger.error("Concurrent tool execution failed: %s", item)
                continue
            tc, result = item
            exec_result.results.append((tc, result))
            if result.success:
                await self._emit(EventType.TOOL_EXECUTED, {
                    "tool": tc.name,
                    "output_length": len(result.output),
                })
            else:
                await self._emit(EventType.TOOL_ERROR, {
                    "tool": tc.name,
                    "error": result.error,
                })

        return exec_result

    async def _emit(self, event_type: EventType, data: dict[str, Any]) -> None:
        if self._event_bus:
            await self._event_bus.emit(AgentEvent(type=event_type, data=data))
