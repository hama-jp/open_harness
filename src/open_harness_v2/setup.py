"""Setup wizard and Ollama auto-start helper for Open Harness v2."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.panel import Panel

console = Console()

# ---------------------------------------------------------------------------
# Provider presets
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "ollama": {
        "label": "Ollama (recommended)",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "api_type": "ollama",
    },
    "lm_studio": {
        "label": "LM Studio",
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        "api_type": "openai",
    },
    "sakura": {
        "label": "Sakura AI Engine (さくらのクラウド)",
        "base_url": "https://api.ai.sakura.ad.jp/v1",
        "api_key": "",
        "api_type": "openai",
    },
    "other": {
        "label": "Other (OpenAI-compatible)",
        "base_url": "",
        "api_key": "",
        "api_type": "openai",
    },
}

_POLICY_PRESETS = {
    "safe": "Strict budgets, confirmation required",
    "balanced": "Reasonable limits (recommended)",
    "full": "No limits",
}


# ---------------------------------------------------------------------------
# Ollama auto-start
# ---------------------------------------------------------------------------

def _is_ollama_reachable(base_url: str) -> bool:
    """Return True if the Ollama server responds at *base_url*."""
    try:
        resp = httpx.get(base_url, timeout=2.0)
        return resp.status_code < 500
    except (httpx.HTTPError, OSError):
        return False


def ensure_ollama(base_url: str = "http://localhost:11434") -> bool:
    """Start the Ollama server if it is not already running.

    Returns True if the server is reachable after this call.
    """
    if _is_ollama_reachable(base_url):
        return True

    ollama_bin = shutil.which("ollama")
    if ollama_bin is None:
        console.print("[yellow]ollama not found on PATH — cannot auto-start.[/yellow]")
        return False

    console.print("[dim]Starting Ollama server …[/dim]")
    try:
        subprocess.Popen(
            [ollama_bin, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        console.print(f"[red]Failed to start Ollama: {exc}[/red]")
        return False

    # Poll for up to 10 seconds
    for _ in range(20):
        time.sleep(0.5)
        if _is_ollama_reachable(base_url):
            console.print("[green]Ollama server started.[/green]")
            return True

    console.print("[yellow]Ollama did not become reachable within 10 s.[/yellow]")
    return False


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _fetch_models(base_url: str, api_key: str, api_type: str = "openai") -> list[str]:
    """Fetch available model names from the provider."""
    try:
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        if api_type == "ollama":
            native_url = base_url.rstrip("/").removesuffix("/v1")
            resp = httpx.get(f"{native_url}/api/tags", headers=headers, timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", []) if "name" in m]
        else:
            resp = httpx.get(f"{base_url}/models", headers=headers, timeout=3.0)
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", []) if "id" in m]
    except Exception:
        return []


def _choose_model(base_url: str, api_key: str, api_type: str = "openai") -> str:
    """Let the user pick a model from the server or type manually."""
    console.print("\n[dim]Checking server for available models …[/dim]")
    models = _fetch_models(base_url, api_key, api_type)

    if models:
        console.print(f"[green]Found {len(models)} model(s):[/green]")
        for i, m in enumerate(models, 1):
            console.print(f"  {i}. {m}")
        console.print(f"  {len(models) + 1}. Enter manually")

        choice = click.prompt(
            "Select model",
            type=click.IntRange(1, len(models) + 1),
            default=1,
        )
        if choice <= len(models):
            return models[choice - 1]
    else:
        console.print(
            "[yellow]Could not fetch models from server "
            "(connection failed or no models loaded).[/yellow]"
        )
        console.print("[dim]You can enter the model name manually.[/dim]")

    return click.prompt("Model name (e.g. qwen3-8b)")


# ---------------------------------------------------------------------------
# YAML builder (v2 ProfileSpec format)
# ---------------------------------------------------------------------------

def _build_yaml(
    provider_key: str,
    base_url: str,
    api_key: str,
    api_type: str,
    model_name: str,
    policy_mode: str,
) -> str:
    """Build a v2 open_harness.yaml string."""
    profile_name = "local"

    lines = [
        "# Open Harness v2 Configuration",
        "# Generated by setup wizard",
        "",
        f"profile: {profile_name}",
        "",
        "profiles:",
        f"  {profile_name}:",
        f"    provider: {provider_key}",
        f"    url: {base_url}",
        f"    api_key: {api_key}",
        f"    api_type: {api_type}",
        "    models:",
        f"      - {model_name}",
        "",
        "policy:",
        f"  mode: {policy_mode}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Wizard
# ---------------------------------------------------------------------------

def run_setup_wizard(config_dir: Path | None = None) -> Path:
    """Run the interactive setup wizard and write open_harness.yaml.

    Returns the path to the generated config file.
    """
    output_dir = config_dir or Path.cwd()

    console.print(Panel(
        "[bold]Open Harness v2 Setup Wizard[/bold]\n"
        "This will create an open_harness.yaml configuration file.",
        border_style="cyan",
    ))

    # Step 1: Provider
    console.print("\n[bold]Step 1:[/bold] LLM Provider")
    provider_choices = list(_PROVIDERS.keys())
    for i, key in enumerate(provider_choices, 1):
        label = _PROVIDERS[key]["label"]
        default_mark = " (default)" if i == 1 else ""
        console.print(f"  {i}. {label}{default_mark}")

    provider_idx = click.prompt(
        "Select provider",
        type=click.IntRange(1, len(provider_choices)),
        default=1,
    )
    provider_key = provider_choices[provider_idx - 1]
    preset = _PROVIDERS[provider_key]
    api_type = preset["api_type"]

    # Step 2: Server URL (with connection test)
    console.print("\n[bold]Step 2:[/bold] Server URL")
    while True:
        if preset["base_url"]:
            base_url: str = click.prompt("Server URL", default=preset["base_url"])
        else:
            base_url = click.prompt("Server URL (e.g. http://localhost:8080/v1)")

        console.print(f"[dim]Testing connection to {base_url} …[/dim]", end=" ")
        try:
            if api_type == "ollama":
                native_url = base_url.rstrip("/").removesuffix("/v1")
                test_resp = httpx.get(native_url, timeout=5.0)
            else:
                test_resp = httpx.get(f"{base_url}/models", timeout=5.0)
            test_resp.raise_for_status()
            console.print("[green]OK[/green]")
            break
        except httpx.ConnectError:
            console.print("[red]FAILED — connection refused[/red]")
            if api_type == "ollama":
                console.print(
                    "[yellow]Ollama server not running.[/yellow]\n"
                    "[dim]Attempting auto-start …[/dim]"
                )
                if ensure_ollama(base_url.rstrip("/").removesuffix("/v1")):
                    break
            else:
                console.print(
                    f"[yellow]Could not connect to {base_url}.[/yellow]\n"
                    "[dim]Check that the server is running and the URL is correct.[/dim]"
                )
        except httpx.TimeoutException:
            console.print("[red]FAILED — timeout[/red]")
        except httpx.HTTPStatusError as exc:
            console.print(f"[yellow]Warning — HTTP {exc.response.status_code}[/yellow]")
            console.print("[dim]Server is reachable but returned an error. This may be OK.[/dim]")
            break
        except httpx.HTTPError as exc:
            console.print(f"[red]FAILED — {exc}[/red]")

        if not click.confirm("Re-enter URL?", default=True):
            console.print("[dim]Continuing with the current URL.[/dim]")
            break

    # Step 3: API key
    console.print("\n[bold]Step 3:[/bold] API Key")
    if preset["api_key"]:
        api_key: str = click.prompt("API key", default=preset["api_key"])
    else:
        api_key = click.prompt("API key", default="no-key")

    # Step 4: Model
    console.print("\n[bold]Step 4:[/bold] Model")
    model_name = _choose_model(base_url, api_key, api_type)
    console.print(f"[green]Selected: {model_name}[/green]")

    # Step 5: Policy
    console.print("\n[bold]Step 5:[/bold] Policy Mode")
    policy_keys = list(_POLICY_PRESETS.keys())
    for i, key in enumerate(policy_keys, 1):
        desc = _POLICY_PRESETS[key]
        console.print(f"  {i}. {key} — {desc}")

    policy_idx = click.prompt(
        "Select policy",
        type=click.IntRange(1, len(policy_keys)),
        default=2,  # balanced
    )
    policy_mode = policy_keys[policy_idx - 1]

    # Step 6: Confirmation
    yaml_content = _build_yaml(
        provider_key=provider_key,
        base_url=base_url,
        api_key=api_key,
        api_type=api_type,
        model_name=model_name,
        policy_mode=policy_mode,
    )

    output_path = output_dir / "open_harness.yaml"
    console.print("\n[bold]Step 6:[/bold] Confirm")
    console.print(f"Config will be written to: [bold]{output_path}[/bold]\n")
    console.print(Panel(yaml_content, title="open_harness.yaml", border_style="dim"))

    if not click.confirm("Write config?", default=True):
        console.print("[yellow]Aborted. No file written.[/yellow]")
        raise SystemExit(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml_content)
    console.print(f"[green]Config written to {output_path}[/green]")
    return output_path
