"""Built-in tools for Open Harness v2."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from open_harness_v2.tools.registry import ToolRegistry


def register_builtins(registry: ToolRegistry) -> None:
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

    for tool_cls in [
        ReadFileTool,
        WriteFileTool,
        EditFileTool,
        ListDirectoryTool,
        SearchFilesTool,
        ShellTool,
        GitStatusTool,
        GitDiffTool,
        GitLogTool,
        GitCommitTool,
        GitBranchTool,
    ]:
        registry.register(tool_cls())
