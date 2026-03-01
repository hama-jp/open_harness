"""Auto-start Ollama server if it's not running."""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

_OLLAMA_DEFAULT_PORT = 11434
_STARTUP_TIMEOUT = 15  # seconds to wait for Ollama to become ready
_POLL_INTERVAL = 0.5


def is_ollama_provider(base_url: str) -> bool:
    """Check if a provider URL looks like an Ollama server."""
    try:
        parsed = urlparse(base_url)
        host = parsed.hostname or ""
        port = parsed.port
        # Match localhost:11434 (default Ollama) or any URL with /v1 on port 11434
        if host in ("localhost", "127.0.0.1", "::1") and port == _OLLAMA_DEFAULT_PORT:
            return True
    except Exception:
        pass
    return False


def is_server_running(base_url: str) -> bool:
    """Check if the Ollama server is reachable."""
    # Use the Ollama native health endpoint (not /v1)
    parsed = urlparse(base_url)
    health_url = f"{parsed.scheme}://{parsed.hostname}:{parsed.port or _OLLAMA_DEFAULT_PORT}"
    try:
        resp = httpx.get(health_url, timeout=3)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        return False


def ensure_ollama_running(base_url: str) -> str | None:
    """Start Ollama server if needed.

    Returns a status message, or None if no action was needed.
    """
    if not is_ollama_provider(base_url):
        return None

    if is_server_running(base_url):
        return None

    # Check if ollama binary exists
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        return "Ollama not found in PATH. Install from https://ollama.com"

    logger.info("Starting Ollama server...")
    try:
        # Start ollama serve in the background
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # detach from parent process
        )
    except OSError as e:
        return f"Failed to start Ollama: {e}"

    # Wait for it to become ready
    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if is_server_running(base_url):
            logger.info("Ollama server is ready")
            return "auto-started"
        time.sleep(_POLL_INTERVAL)

    return f"Ollama started but not ready after {_STARTUP_TIMEOUT}s"
