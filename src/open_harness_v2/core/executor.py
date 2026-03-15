"""Executor — runs tool calls with policy checks, approval, hooks, and sandbox.

Integrates:
- Policy engine (budget/path guardrails)
- Approval system (interactive human-in-the-loop, from Codex)
- Hook decision control (pre-tool allow/deny/ask, from Claude Code)
- Sandbox wrapping (OS-level isolation for shell commands, from Codex)

Supports both sequential and concurrent tool execution.
"""

from __future__ import annotations

import asyncio
import logging
import time
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
    """Runs tool calls through policy checks, approval, hooks, and the tool registry.

    Usage::

        executor = Executor(registry, policy, event_bus)
        result = await executor.execute(tool_calls)
    """

    def __init__(
        self,
        registry: ToolRegistry,
        policy: PolicyEngine | None = None,
        event_bus: EventBus | None = None,
        approval_engine: Any = None,  # ApprovalEngine
        hook_engine: Any = None,  # HookEngine
        sandbox_engine: Any = None,  # SandboxEngine
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._event_bus = event_bus
        self._approval = approval_engine
        self._hook_engine = hook_engine
        self._sandbox = sandbox_engine

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

    async def _check_pre_execution(
        self, tc: ToolCall
    ) -> ToolResult | None:
        """Run pre-execution checks (policy, hooks, approval).

        Returns a ToolResult if the call should be blocked, or None to proceed.
        """
        # 1. Policy check
        if self._policy:
            violation = self._policy.check(tc.name, tc.arguments)
            if violation:
                await self._emit(EventType.POLICY_VIOLATION, {
                    "tool": tc.name,
                    "rule": violation.rule,
                    "message": violation.message,
                })
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Policy violation: {violation.message}",
                )

        # 2. Hook decision control (pre-tool hooks)
        if self._hook_engine:
            from open_harness_v2.hooks.engine import HookDecision
            hook_result = await self._hook_engine.check_pre_tool(
                tc.name, tc.arguments
            )
            if hook_result.decision == HookDecision.DENY:
                _logger.info(
                    "Hook denied tool call: %s — %s",
                    tc.name, hook_result.message,
                )
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Hook denied: {hook_result.message}",
                )
            elif hook_result.decision == HookDecision.ASK:
                # Hook wants user confirmation — delegate to approval
                if self._approval:
                    from open_harness_v2.approval import (
                        ApprovalDecision,
                        ApprovalRequest,
                    )
                    request = ApprovalRequest(
                        tool_name=tc.name,
                        arguments=tc.arguments,
                        reason=hook_result.message or f"Hook requires confirmation for {tc.name}",
                        category="hook",
                    )
                    decision = await self._approval.request_approval(request)
                    if decision == ApprovalDecision.DENIED:
                        return ToolResult(
                            success=False,
                            output="",
                            error="User denied hook-requested approval",
                        )
                else:
                    _logger.warning(
                        "Hook requested user confirmation but no approval "
                        "engine available; allowing: %s",
                        tc.name,
                    )

        # 3. Approval check
        if self._approval:
            from open_harness_v2.approval import ApprovalDecision
            request = self._approval.needs_approval(tc.name, tc.arguments)
            if request:
                decision = await self._approval.request_approval(request)
                if decision == ApprovalDecision.DENIED:
                    return ToolResult(
                        success=False,
                        output="",
                        error="User denied approval for this action",
                    )

        return None  # All checks passed

    def _maybe_sandbox_args(self, tc: ToolCall) -> dict[str, Any]:
        """Apply sandbox wrapping to shell tool arguments if applicable."""
        if not self._sandbox or tc.name != "shell":
            return tc.arguments

        if not self._sandbox.is_restricted:
            return tc.arguments

        command = tc.arguments.get("command", "")
        if not command:
            return tc.arguments

        wrapped = self._sandbox.wrap_command(command)
        if wrapped != command:
            _logger.debug("Sandbox wrapped command: %s", wrapped[:100])
            return {**tc.arguments, "command": wrapped}

        return tc.arguments

    async def _execute_sequential(
        self, tool_calls: list[ToolCall]
    ) -> ExecutionResult:
        exec_result = ExecutionResult()
        for tc in tool_calls:
            # Pre-execution checks (policy + hooks + approval)
            block_result = await self._check_pre_execution(tc)
            if block_result:
                if self._policy:
                    # Check if it was a policy violation for tracking
                    violation = self._policy.check(tc.name, tc.arguments)
                    if violation:
                        exec_result.violations.append((tc, violation))
                exec_result.results.append((tc, block_result))
                continue

            # Apply sandbox wrapping for shell commands
            effective_args = self._maybe_sandbox_args(tc)

            # Execute
            await self._emit(EventType.TOOL_EXECUTING, {
                "tool": tc.name,
                "args": tc.arguments,  # Show original args in events
            })

            start = time.monotonic()
            result = await self._registry.execute(tc.name, effective_args)
            elapsed_ms = (time.monotonic() - start) * 1000

            # Record in policy budget
            if self._policy:
                self._policy.record(tc.name)

            if result.success:
                await self._emit(EventType.TOOL_EXECUTED, {
                    "tool": tc.name,
                    "success": True,
                    "output": result.output,
                    "output_length": len(result.output),
                    "elapsed_ms": elapsed_ms,
                })
            else:
                await self._emit(EventType.TOOL_ERROR, {
                    "tool": tc.name,
                    "success": False,
                    "error": result.error,
                    "elapsed_ms": elapsed_ms,
                })

            exec_result.results.append((tc, result))

        return exec_result

    async def _execute_concurrent(
        self, tool_calls: list[ToolCall]
    ) -> ExecutionResult:
        """Execute all tool calls concurrently."""
        exec_result = ExecutionResult()

        # First, check all pre-execution (policies + hooks + approval)
        to_run: list[ToolCall] = []
        for tc in tool_calls:
            block_result = await self._check_pre_execution(tc)
            if block_result:
                if self._policy:
                    violation = self._policy.check(tc.name, tc.arguments)
                    if violation:
                        exec_result.violations.append((tc, violation))
                exec_result.results.append((tc, block_result))
                continue
            to_run.append(tc)

        if not to_run:
            return exec_result

        # Execute concurrently
        async def _run_one(tc: ToolCall) -> tuple[ToolCall, ToolResult, float]:
            effective_args = self._maybe_sandbox_args(tc)
            await self._emit(EventType.TOOL_EXECUTING, {
                "tool": tc.name,
                "args": tc.arguments,
            })
            start = time.monotonic()
            result = await self._registry.execute(tc.name, effective_args)
            elapsed_ms = (time.monotonic() - start) * 1000
            if self._policy:
                self._policy.record(tc.name)
            return tc, result, elapsed_ms

        pairs = await asyncio.gather(
            *[_run_one(tc) for tc in to_run],
            return_exceptions=True,
        )

        for i, item in enumerate(pairs):
            if isinstance(item, Exception):
                _logger.error("Concurrent tool execution failed: %s", item)
                tc = to_run[i]
                error_result = ToolResult(
                    success=False,
                    output="",
                    error=f"Execution exception: {type(item).__name__}: {item}",
                )
                exec_result.results.append((tc, error_result))
                await self._emit(EventType.TOOL_ERROR, {
                    "tool": tc.name,
                    "success": False,
                    "error": error_result.error,
                })
                continue
            tc, result, elapsed_ms = item
            exec_result.results.append((tc, result))
            if result.success:
                await self._emit(EventType.TOOL_EXECUTED, {
                    "tool": tc.name,
                    "success": True,
                    "output": result.output,
                    "output_length": len(result.output),
                    "elapsed_ms": elapsed_ms,
                })
            else:
                await self._emit(EventType.TOOL_ERROR, {
                    "tool": tc.name,
                    "success": False,
                    "error": result.error,
                    "elapsed_ms": elapsed_ms,
                })

        return exec_result

    async def _emit(self, event_type: EventType, data: dict[str, Any]) -> None:
        if self._event_bus:
            await self._event_bus.emit(AgentEvent(type=event_type, data=data))
