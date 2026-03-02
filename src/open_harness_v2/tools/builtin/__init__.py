"""Built-in tools for Open Harness v2."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from open_harness_v2.memory.project import ProjectMemory
    from open_harness_v2.tools.registry import ToolRegistry


def register_builtins(
    registry: ToolRegistry,
    project_memory: ProjectMemory | None = None,
) -> None:
    """Register all built-in tools with the given registry."""
    from open_harness_v2.tools.builtin.file_ops import (
        EditFileTool,
        ListDirectoryTool,
        ReadFileTool,
        SearchFilesTool,
        WriteFileTool,
    )
    from open_harness_v2.tools.builtin.git_tools import (
        GitBranchTool,
        GitCommitTool,
        GitDiffTool,
        GitLogTool,
        GitStatusTool,
    )
    from open_harness_v2.tools.builtin.shell import ShellTool
    from open_harness_v2.tools.builtin.testing import TestRunnerTool
    from open_harness_v2.tools.builtin.external import (
        ClaudeCodeTool,
        CodexTool,
        GeminiCliTool,
    )

    for tool_cls in [
        ReadFileTool,
        WriteFileTool,
        EditFileTool,
        ListDirectoryTool,
        SearchFilesTool,
        ShellTool,
        TestRunnerTool,
        GitStatusTool,
        GitDiffTool,
        GitLogTool,
        GitCommitTool,
        GitBranchTool,
        CodexTool,
        ClaudeCodeTool,
        GeminiCliTool,
    ]:
        registry.register(tool_cls())

    # Memory tools (require a ProjectMemory instance)
    if project_memory is not None:
        from open_harness_v2.tools.builtin.memory_tool import (
            ForgetTool,
            RememberTool,
        )

        registry.register(RememberTool(project_memory))
        registry.register(ForgetTool(project_memory))
