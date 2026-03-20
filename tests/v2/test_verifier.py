"""Tests for the Verifier — goal completion validation."""

import pytest

from open_harness_v2.core.verifier import (
    VerificationResult,
    VerificationStatus,
    Verifier,
)


class TestVerifierBasics:
    """Basic Verifier functionality."""

    async def test_no_actions_fails(self):
        v = Verifier()
        result = await v.verify("Fix the bug", [])
        assert result.status in (
            VerificationStatus.FAILED,
            VerificationStatus.INCONCLUSIVE,
        )
        assert not result.is_acceptable

    async def test_all_succeeded_passes(self):
        v = Verifier()
        actions = [
            {"tool_name": "read_file", "success": True, "output": "content"},
            {"tool_name": "edit_file", "success": True, "output": "ok"},
            {"tool_name": "shell", "success": True, "output": "tests passed"},
        ]
        result = await v.verify(
            "Fix the bug",
            actions,
            files_modified={"src/main.py"},
        )
        assert result.status in (VerificationStatus.PASSED, VerificationStatus.PARTIAL)

    async def test_low_success_rate_fails(self):
        v = Verifier()
        actions = [
            {"tool_name": "shell", "success": False, "output": ""},
            {"tool_name": "shell", "success": False, "output": ""},
            {"tool_name": "read_file", "success": True, "output": "x"},
        ]
        result = await v.verify("Fix the bug", actions)
        assert "success rate" in " ".join(result.checks_failed).lower() or \
               result.status != VerificationStatus.PASSED

    async def test_code_task_no_files_fails(self):
        v = Verifier()
        actions = [
            {"tool_name": "read_file", "success": True, "output": "content"},
        ]
        result = await v.verify("Implement the feature", actions, files_modified=set())
        assert any("no files" in c.lower() for c in result.checks_failed)

    async def test_non_code_task_no_files_ok(self):
        v = Verifier()
        actions = [
            {"tool_name": "read_file", "success": True, "output": "content"},
        ]
        result = await v.verify("Explain how auth works", actions, files_modified=set())
        # Should not fail for non-code task
        assert not any("no files" in c.lower() for c in result.checks_failed)


class TestVerifierTests:
    """Tests for test result checking."""

    async def test_all_tests_pass(self):
        v = Verifier()
        actions = [{"tool_name": "shell", "success": True, "output": "ok"}]
        result = await v.verify(
            "Fix bug",
            actions,
            files_modified={"a.py"},
            test_results={"passed": 10, "failed": 0},
        )
        assert any("tests passed" in c.lower() for c in result.checks_passed)

    async def test_some_tests_fail(self):
        v = Verifier()
        actions = [{"tool_name": "shell", "success": True, "output": "ok"}]
        result = await v.verify(
            "Fix bug",
            actions,
            files_modified={"a.py"},
            test_results={"passed": 8, "failed": 2},
        )
        assert any("tests failed" in c.lower() for c in result.checks_failed)


class TestQuickVerify:
    """Tests for lightweight quick_verify."""

    async def test_good_success_rate(self):
        v = Verifier()
        result = await v.quick_verify("Fix bug", "Done!", 0.9, 2)
        assert result.status == VerificationStatus.PASSED

    async def test_low_success_rate(self):
        v = Verifier()
        result = await v.quick_verify("Fix bug", "Done!", 0.2, 0)
        assert result.status != VerificationStatus.PASSED

    async def test_code_task_no_files(self):
        v = Verifier()
        result = await v.quick_verify("Implement feature", "Done!", 0.9, 0)
        assert result.status != VerificationStatus.PASSED


class TestLLMVerification:
    """Tests for LLM-based verification."""

    async def test_llm_verify_passed(self):
        async def mock_llm(messages):
            return '{"passed": true, "reason": "Goal achieved", "suggestion": ""}'

        v = Verifier(llm_call=mock_llm)
        actions = [
            {"tool_name": "edit_file", "success": True, "output": "ok"},
        ]
        result = await v.verify("Fix bug", actions, files_modified={"a.py"})
        assert any("llm" in c.lower() for c in result.checks_passed)

    async def test_llm_verify_failed(self):
        async def mock_llm(messages):
            return '{"passed": false, "reason": "Tests not run", "suggestion": "Run tests"}'

        v = Verifier(llm_call=mock_llm)
        actions = [
            {"tool_name": "edit_file", "success": True, "output": "ok"},
        ]
        result = await v.verify("Fix bug", actions, files_modified={"a.py"})
        assert any("llm" in c.lower() for c in result.checks_failed)

    async def test_llm_verify_error_fallback(self):
        async def broken_llm(messages):
            raise RuntimeError("LLM down")

        v = Verifier(llm_call=broken_llm)
        actions = [
            {"tool_name": "edit_file", "success": True, "output": "ok"},
        ]
        # Should not crash, just skip LLM check
        result = await v.verify("Fix bug", actions, files_modified={"a.py"})
        assert result.status is not None


class TestVerificationResult:
    """Tests for VerificationResult properties."""

    def test_is_acceptable_passed(self):
        r = VerificationResult(
            status=VerificationStatus.PASSED, confidence=0.8, summary="ok"
        )
        assert r.is_acceptable

    def test_is_acceptable_skipped(self):
        r = VerificationResult(
            status=VerificationStatus.SKIPPED, confidence=0.5, summary="n/a"
        )
        assert r.is_acceptable

    def test_not_acceptable_failed(self):
        r = VerificationResult(
            status=VerificationStatus.FAILED, confidence=0.8, summary="bad"
        )
        assert not r.is_acceptable
