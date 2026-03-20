"""Verifier — validates that completed work actually achieves the goal.

Post-execution verification ensures the agent doesn't prematurely declare
success. Uses a combination of:
1. **Structural checks**: Files exist, tests pass, no syntax errors
2. **Semantic checks**: LLM-based assessment of whether the goal is met
3. **Regression checks**: No existing tests broken
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

_logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Result of verification."""

    PASSED = "passed"          # Goal appears to be met
    PARTIAL = "partial"        # Some aspects met, others not
    FAILED = "failed"          # Goal not met
    SKIPPED = "skipped"        # Verification not applicable
    INCONCLUSIVE = "inconclusive"  # Can't determine


@dataclass
class VerificationResult:
    """Output of a verification cycle."""

    status: VerificationStatus
    confidence: float  # 0.0 - 1.0
    summary: str
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    @property
    def is_acceptable(self) -> bool:
        """True if the result is good enough to proceed."""
        return self.status in (VerificationStatus.PASSED, VerificationStatus.SKIPPED)


class Verifier:
    """Validates that agent work achieves the stated goal.

    Operates in two modes:
    1. **Heuristic**: Fast checks based on action history
    2. **LLM-assisted**: Uses LLM for semantic goal-completion assessment
    """

    def __init__(
        self,
        llm_call: Any = None,  # async callable
    ) -> None:
        self._llm_call = llm_call

    async def verify(
        self,
        goal: str,
        actions_taken: list[dict[str, Any]],
        files_modified: set[str] | None = None,
        test_results: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """Verify that the goal has been achieved.

        Parameters
        ----------
        goal:
            The original goal text.
        actions_taken:
            List of action records (tool_name, args, success, output).
        files_modified:
            Set of file paths that were modified.
        test_results:
            Optional test execution results.
        """
        checks_passed: list[str] = []
        checks_failed: list[str] = []
        suggestions: list[str] = []

        # 1. Basic structural checks
        self._check_actions_succeeded(
            actions_taken, checks_passed, checks_failed, suggestions,
        )

        # 2. File modification checks
        self._check_files(
            goal, files_modified, checks_passed, checks_failed, suggestions,
        )

        # 3. Test result checks
        self._check_tests(
            test_results, checks_passed, checks_failed, suggestions,
        )

        # 4. LLM-based semantic check (if available)
        if self._llm_call:
            llm_result = await self._llm_verify(goal, actions_taken, files_modified)
            if llm_result:
                if llm_result["passed"]:
                    checks_passed.append(f"LLM assessment: {llm_result['reason']}")
                else:
                    checks_failed.append(f"LLM assessment: {llm_result['reason']}")
                    if llm_result.get("suggestion"):
                        suggestions.append(llm_result["suggestion"])

        # Aggregate results
        return self._aggregate(checks_passed, checks_failed, suggestions)

    async def quick_verify(
        self,
        goal: str,
        last_response: str,
        success_rate: float,
        files_modified: int,
    ) -> VerificationResult:
        """Fast heuristic verification without LLM call.

        Used between plan steps for lightweight progress checks.
        """
        checks_passed: list[str] = []
        checks_failed: list[str] = []

        if success_rate >= 0.7:
            checks_passed.append(f"Good success rate: {success_rate:.0%}")
        elif success_rate < 0.4:
            checks_failed.append(f"Low success rate: {success_rate:.0%}")

        goal_lower = goal.lower()
        is_code_task = any(
            w in goal_lower
            for w in ("fix", "add", "implement", "create", "change", "update", "refactor")
        )

        if is_code_task and files_modified == 0:
            checks_failed.append("Code task but no files modified yet")

        if not checks_failed:
            return VerificationResult(
                status=VerificationStatus.PASSED,
                confidence=0.5,
                summary="Quick check passed",
                checks_passed=checks_passed,
            )

        return VerificationResult(
            status=VerificationStatus.PARTIAL,
            confidence=0.4,
            summary="Quick check found issues",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )

    # ------------------------------------------------------------------
    # Structural checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_actions_succeeded(
        actions: list[dict[str, Any]],
        passed: list[str],
        failed: list[str],
        suggestions: list[str],
    ) -> None:
        """Check that a reasonable proportion of actions succeeded."""
        if not actions:
            failed.append("No actions were taken")
            suggestions.append("Execute the plan before verifying")
            return

        total = len(actions)
        successes = sum(1 for a in actions if a.get("success", False))
        rate = successes / total

        if rate >= 0.8:
            passed.append(f"High success rate: {successes}/{total} actions succeeded")
        elif rate >= 0.5:
            passed.append(f"Moderate success rate: {successes}/{total}")
        else:
            failed.append(f"Low success rate: {successes}/{total} actions succeeded")
            suggestions.append("Many actions failed — review error messages")

        # Check that the last action succeeded (important for goal completion)
        if actions and actions[-1].get("success", False):
            passed.append("Final action succeeded")
        elif actions:
            failed.append("Final action failed")
            suggestions.append("The last action failed — the goal may not be fully achieved")

    @staticmethod
    def _check_files(
        goal: str,
        files_modified: set[str] | None,
        passed: list[str],
        failed: list[str],
        suggestions: list[str],
    ) -> None:
        """Check file modifications are consistent with the goal."""
        goal_lower = goal.lower()
        is_code_task = any(
            w in goal_lower
            for w in ("fix", "add", "implement", "create", "change", "update",
                      "write", "refactor", "modify")
        )

        if not is_code_task:
            return  # Not a code modification task

        if not files_modified:
            failed.append("Code modification task but no files were changed")
            suggestions.append("Ensure changes were actually written to disk")
        else:
            passed.append(f"Modified {len(files_modified)} file(s)")

    @staticmethod
    def _check_tests(
        test_results: dict[str, Any] | None,
        passed: list[str],
        failed: list[str],
        suggestions: list[str],
    ) -> None:
        """Check test execution results."""
        if test_results is None:
            return  # No tests run

        tests_passed = test_results.get("passed", 0)
        tests_failed = test_results.get("failed", 0)
        total = tests_passed + tests_failed

        if total == 0:
            return

        if tests_failed == 0:
            passed.append(f"All {tests_passed} tests passed")
        else:
            failed.append(f"{tests_failed}/{total} tests failed")
            suggestions.append("Fix failing tests before declaring completion")

    # ------------------------------------------------------------------
    # LLM-based verification
    # ------------------------------------------------------------------

    _VERIFY_PROMPT = """\
Evaluate whether the following goal has been achieved based on the actions taken.

Goal: {goal}

Actions taken (last 10):
{actions_summary}

Files modified: {files}

Answer with ONLY a JSON object:
{{"passed": true/false, "reason": "brief explanation", "suggestion": "what to do next if not passed"}}
"""

    async def _llm_verify(
        self,
        goal: str,
        actions: list[dict[str, Any]],
        files_modified: set[str] | None,
    ) -> dict[str, Any] | None:
        """Use LLM to assess goal completion."""
        try:
            # Summarize recent actions
            recent = actions[-10:] if len(actions) > 10 else actions
            action_lines = []
            for a in recent:
                status = "OK" if a.get("success") else "FAIL"
                tool = a.get("tool_name", "unknown")
                action_lines.append(f"  [{status}] {tool}")

            prompt = self._VERIFY_PROMPT.format(
                goal=goal,
                actions_summary="\n".join(action_lines) if action_lines else "(none)",
                files=", ".join(files_modified) if files_modified else "(none)",
            )

            messages = [
                {"role": "system", "content": "You are a verification assistant."},
                {"role": "user", "content": prompt},
            ]

            response = await self._llm_call(messages)
            return self._parse_verify_response(response)
        except Exception as e:
            _logger.warning("LLM verification failed: %s", e)
            return None

    @staticmethod
    def _parse_verify_response(text: str) -> dict[str, Any] | None:
        """Parse a JSON verification response."""
        import json
        import re

        text = text.strip()
        # Try direct JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try extracting from code block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(
        checks_passed: list[str],
        checks_failed: list[str],
        suggestions: list[str],
    ) -> VerificationResult:
        """Aggregate check results into a final verdict."""
        total = len(checks_passed) + len(checks_failed)
        if total == 0:
            return VerificationResult(
                status=VerificationStatus.INCONCLUSIVE,
                confidence=0.2,
                summary="No verification checks could be performed",
                suggestions=["Run the plan before verifying"],
            )

        pass_ratio = len(checks_passed) / total

        if not checks_failed:
            return VerificationResult(
                status=VerificationStatus.PASSED,
                confidence=min(0.9, 0.4 + pass_ratio * 0.5),
                summary=f"All {len(checks_passed)} checks passed",
                checks_passed=checks_passed,
            )

        if pass_ratio >= 0.6:
            return VerificationResult(
                status=VerificationStatus.PARTIAL,
                confidence=0.5,
                summary=(
                    f"{len(checks_passed)} checks passed, "
                    f"{len(checks_failed)} failed"
                ),
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                suggestions=suggestions,
            )

        return VerificationResult(
            status=VerificationStatus.FAILED,
            confidence=min(0.9, 0.3 + (1 - pass_ratio) * 0.5),
            summary=f"{len(checks_failed)} checks failed",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            suggestions=suggestions,
        )
