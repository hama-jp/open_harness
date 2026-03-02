"""Self-update: git pull + reinstall."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from open_harness_v2 import __version__

# Repository root — two levels up from this file (src/open_harness_v2/update.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def self_update(console: Console) -> bool:
    """Pull latest code from git and reinstall the package.

    Returns True if an update was applied, False if already up-to-date.
    """
    repo = _REPO_ROOT
    git_dir = repo / ".git"
    if not git_dir.is_dir():
        console.print(f"[red]Not a git repository: {repo}[/red]")
        return False

    console.print(f"[dim]Current version: v{__version__}[/dim]")
    console.print(f"[dim]Updating from {repo} ...[/dim]")

    # 1. git fetch
    fetch = subprocess.run(
        ["git", "fetch"], cwd=repo, capture_output=True, text=True, timeout=30,
    )
    if fetch.returncode != 0:
        console.print(f"[red]git fetch failed: {fetch.stderr.strip()}[/red]")
        return False

    # Compare local HEAD vs remote
    local = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
    )
    remote = subprocess.run(
        ["git", "rev-parse", "@{u}"], cwd=repo, capture_output=True, text=True,
    )

    no_upstream = remote.returncode != 0
    if no_upstream:
        console.print(
            "[yellow]No upstream branch configured. Will pull origin/main.[/yellow]"
        )
    elif local.stdout.strip() == remote.stdout.strip():
        console.print(f"[green]Already up-to-date (v{__version__}).[/green]")
        return False

    # 2. Stash local changes so pull doesn't fail on dirty working tree
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo, capture_output=True, text=True,
    )
    stashed = False
    if status.stdout.strip():
        console.print("[dim]Stashing local changes ...[/dim]")
        stash = subprocess.run(
            ["git", "stash", "push", "-m", "open-harness: pre-update stash"],
            cwd=repo, capture_output=True, text=True, timeout=15,
        )
        if stash.returncode == 0 and "No local changes" not in stash.stdout:
            stashed = True

    # 3. git pull — use explicit remote/branch when no upstream is set
    if no_upstream:
        default_branch = "main"
        for candidate in ("main", "master"):
            check = subprocess.run(
                ["git", "rev-parse", "--verify", f"origin/{candidate}"],
                cwd=repo, capture_output=True, text=True,
            )
            if check.returncode == 0:
                default_branch = candidate
                break
        console.print(f"[dim]git pull origin {default_branch} ...[/dim]")
        pull = subprocess.run(
            ["git", "pull", "--ff-only", "origin", default_branch],
            cwd=repo, capture_output=True, text=True, timeout=60,
        )
    else:
        console.print("[dim]git pull ...[/dim]")
        pull = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=repo, capture_output=True, text=True, timeout=60,
        )

    if pull.returncode != 0:
        console.print(f"[red]git pull failed: {pull.stderr.strip()}[/red]")
        if stashed:
            subprocess.run(
                ["git", "stash", "pop"],
                cwd=repo, capture_output=True, text=True,
            )
            console.print("[dim]Restored stashed changes.[/dim]")
        else:
            console.print("[dim]Hint: commit or stash local changes first.[/dim]")
        return False

    if pull.stdout.strip():
        console.print(f"[dim]{pull.stdout.strip()}[/dim]")

    # Restore stashed changes after successful pull
    if stashed:
        pop = subprocess.run(
            ["git", "stash", "pop"],
            cwd=repo, capture_output=True, text=True,
        )
        if pop.returncode == 0:
            console.print("[dim]Restored stashed changes.[/dim]")
        else:
            console.print(
                f"[yellow]Stash pop failed: {pop.stderr.strip()[:100]}[/yellow]"
            )
            console.print("[dim]Your changes are still in git stash.[/dim]")

    # 4. Reinstall package (prefer uv, fall back to pip)
    if shutil.which("uv"):
        install_cmd = ["uv", "pip", "install", "-e", str(repo), "-q"]
        install_label = "uv pip install -e ."
    else:
        install_cmd = [sys.executable, "-m", "pip", "install", "-e", str(repo), "-q"]
        install_label = "pip install -e ."
    console.print(f"[dim]{install_label} ...[/dim]")
    pip = subprocess.run(
        install_cmd, cwd=repo, capture_output=True, text=True, timeout=120,
    )
    if pip.returncode != 0:
        console.print(
            f"[yellow]{install_label} warning: {pip.stderr.strip()[:200]}[/yellow]"
        )

    # Re-read version from the updated package
    # (the running process still has the old version in memory)
    new_ver_proc = subprocess.run(
        [sys.executable, "-c", "from open_harness_v2 import __version__; print(__version__)"],
        cwd=repo, capture_output=True, text=True,
    )
    new_ver = new_ver_proc.stdout.strip() if new_ver_proc.returncode == 0 else "?"

    console.print(f"[green]Updated: v{__version__} -> v{new_ver}[/green]")
    console.print("[dim]Please restart harness to apply changes.[/dim]")
    return True
