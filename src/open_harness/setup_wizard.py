"""Interactive setup wizard for generating open_harness.yaml."""

from __future__ import annotations

import shutil
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.panel import Panel

console = Console()

# Provider presets
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
    "other": {
        "label": "Other (OpenAI-compatible)",
        "base_url": "",
        "api_key": "",
        "api_type": "openai",
    },
}

# External agents to auto-detect
_EXTERNAL_AGENTS = [
    ("claude", "claude", "Complex refactoring, planning"),
    ("codex", "codex", "Fast coding and code review"),
    ("gemini", "gemini", "Large codebase understanding"),
]

# Policy presets
_POLICY_PRESETS = {
    "safe": "Strict budgets, confirmation required",
    "balanced": "Reasonable limits (recommended)",
    "full": "No limits",
}


def _fetch_models(base_url: str, api_key: str, api_type: str = "openai") -> list[str]:
    """Try to fetch model list from the provider."""
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if api_type == "ollama":
            # Use Ollama native /api/tags
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
    """Let user pick a model from the server or type manually."""
    console.print("\n[dim]Checking server for available models...[/dim]")
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
        console.print("[yellow]Could not fetch models from server (connection failed or no models loaded).[/yellow]")
        console.print("[dim]You can enter the model name manually.[/dim]")

    return click.prompt("Model name (e.g. unsloth/qwen3.5-35b-a3b@q4_k_m)")


def _detect_agents() -> list[tuple[str, str, str, bool]]:
    """Detect external agents on PATH.

    Returns list of (key, command, description, found).
    """
    results = []
    for key, cmd, desc in _EXTERNAL_AGENTS:
        path = shutil.which(cmd)
        found = path is not None
        location = f" (found at {path})" if found else " (not found)"
        results.append((key, cmd, desc + location, found))
    return results


def _build_yaml(
    provider_key: str,
    base_url: str,
    api_key: str,
    api_type: str,
    model_name: str,
    agents: dict[str, bool],
    policy_mode: str,
) -> str:
    """Build open_harness.yaml content as a formatted string with comments."""
    is_ollama = (api_type == "ollama")

    # External agents section
    agent_lines = []
    for key, cmd, _ in _EXTERNAL_AGENTS:
        enabled = "true" if agents.get(key, False) else "false"
        agent_lines.append(f"  {key}:")
        agent_lines.append(f"    enabled: {enabled}")
        agent_lines.append(f'    command: "{cmd}"')
    agents_block = "\n".join(agent_lines)

    # Provider block (Ollama gets extra fields)
    if is_ollama:
        provider_block = (
            f'    {provider_key}:\n'
            f'      base_url: "{base_url}"\n'
            f'      api_key: "{api_key}"\n'
            f'      api_type: "ollama"  # use native /api/chat (supports think: false)\n'
            f'      extra_params:\n'
            f'        think: false  # disable thinking for faster responses'
        )
    else:
        provider_block = (
            f'    {provider_key}:\n'
            f'      base_url: "{base_url}"\n'
            f'      api_key: "{api_key}"'
        )

    # Model tiers (Ollama gets context_length)
    if is_ollama:
        models_block = f"""\
    small:
      provider: "{provider_key}"
      model: "{model_name}"
      max_tokens: 4096
      context_length: 32768
      description: "Fast, simple tasks (32K ctx)"

    medium:
      provider: "{provider_key}"
      model: "{model_name}"
      max_tokens: 8192
      context_length: 49152
      description: "Balanced performance (48K ctx)"

    large:
      provider: "{provider_key}"
      model: "{model_name}"
      max_tokens: 16384
      context_length: 65536
      description: "Complex reasoning (64K ctx)\""""
    else:
        models_block = f"""\
    small:
      provider: "{provider_key}"
      model: "{model_name}"
      max_tokens: 4096
      description: "Fast, simple tasks"

    medium:
      provider: "{provider_key}"
      model: "{model_name}"
      max_tokens: 8192
      description: "Balanced performance"

    large:
      provider: "{provider_key}"
      model: "{model_name}"
      max_tokens: 16384
      description: "Complex reasoning\""""

    thinking_mode = "never" if is_ollama else "auto"

    return f"""\
# Open Harness Configuration
# Generated by setup wizard

llm:
  default_provider: "{provider_key}"

  providers:
{provider_block}

  # Model tiers for routing (small -> medium -> large)
  models:
{models_block}

  default_tier: "medium"

# Compensation engine settings
compensation:
  max_retries: 3
  retry_strategies:
    - "refine_prompt"
    - "add_examples"
    - "escalate_model"
  parse_fallback: true
  thinking_mode: "{thinking_mode}"

# Tool settings
tools:
  shell:
    timeout: 30
    allowed_commands: []
    blocked_commands: ["rm -rf /", "mkfs", "dd if="]
  file:
    max_read_size: 100000

# External agents
external_agents:
{agents_block}

# Execution policy
policy:
  mode: "{policy_mode}"

# Memory
memory:
  backend: "sqlite"
  db_path: "~/.open_harness/memory.db"
  max_conversation_turns: 50
"""


def run_setup_wizard(config_dir: Path | None = None) -> Path:
    """Run interactive setup wizard and write open_harness.yaml.

    Args:
        config_dir: Directory to write the config file to.
                    Defaults to current working directory.

    Returns:
        Path to the generated config file.
    """
    output_dir = config_dir or Path.cwd()

    console.print(Panel(
        "[bold]Open Harness Setup Wizard[/bold]\n"
        "This will create an open_harness.yaml configuration file.",
        border_style="cyan",
    ))

    # Step 1: Provider selection
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
    api_type = preset.get("api_type", "openai")

    # Step 2: Server URL (with connection test)
    console.print(f"\n[bold]Step 2:[/bold] Server URL")
    while True:
        if preset["base_url"]:
            base_url = click.prompt("Server URL", default=preset["base_url"])
        else:
            base_url = click.prompt("Server URL (e.g. http://localhost:8080/v1)")

        # Connection test — use native endpoint for Ollama
        console.print(f"[dim]Testing connection to {base_url} ...[/dim]", end=" ")
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
                    "[dim]Start with: ollama serve[/dim]\n"
                    "[dim]Harness will also auto-start it at runtime.[/dim]"
                )
            else:
                console.print(
                    f"[yellow]Could not connect to {base_url}.[/yellow]\n"
                    "[dim]Check that the server is running and the URL is correct.[/dim]"
                )
        except httpx.TimeoutException:
            console.print("[red]FAILED — timeout[/red]")
            console.print(
                f"[yellow]Connection to {base_url} timed out.[/yellow]\n"
                "[dim]The server may be slow to respond or unreachable.[/dim]"
            )
        except httpx.HTTPStatusError as e:
            # Server responded but with an error — still reachable
            console.print(f"[yellow]Warning — HTTP {e.response.status_code}[/yellow]")
            console.print("[dim]Server is reachable but returned an error. This may be OK.[/dim]")
            break
        except httpx.HTTPError as e:
            console.print(f"[red]FAILED — {e}[/red]")

        if not click.confirm("Re-enter URL?", default=True):
            console.print("[dim]Continuing with the current URL.[/dim]")
            break

    # Step 3: API key
    console.print(f"\n[bold]Step 3:[/bold] API Key")
    if preset["api_key"]:
        api_key = click.prompt("API key", default=preset["api_key"])
    else:
        api_key = click.prompt("API key", default="no-key")

    # Step 4: Model
    console.print(f"\n[bold]Step 4:[/bold] Model")
    model_name = _choose_model(base_url, api_key, api_type)
    console.print(f"[green]Selected: {model_name}[/green]")

    # Step 5: External agents
    console.print(f"\n[bold]Step 5:[/bold] External Agents")
    detected = _detect_agents()
    any_found = any(found for _, _, _, found in detected)
    agents: dict[str, bool] = {}

    if any_found:
        console.print("Auto-detected agents:")
        for key, cmd, desc, found in detected:
            icon = "[green]v[/green]" if found else "[red]x[/red]"
            console.print(f"  [{icon}] {cmd}: {desc}")
        enable_agents = click.confirm("Enable detected agents?", default=True)
        for key, _, _, found in detected:
            agents[key] = found and enable_agents
    else:
        console.print("[dim]No external agents found on PATH.[/dim]")
        for key, _, _, _ in detected:
            agents[key] = False

    # Step 6: Policy mode
    console.print(f"\n[bold]Step 6:[/bold] Policy Mode")
    policy_keys = list(_POLICY_PRESETS.keys())
    for i, key in enumerate(policy_keys, 1):
        desc = _POLICY_PRESETS[key]
        console.print(f"  {i}. {key} - {desc}")

    policy_idx = click.prompt(
        "Select policy",
        type=click.IntRange(1, len(policy_keys)),
        default=2,  # balanced
    )
    policy_mode = policy_keys[policy_idx - 1]

    # Step 7: Confirmation
    yaml_content = _build_yaml(
        provider_key=provider_key,
        base_url=base_url,
        api_key=api_key,
        api_type=api_type,
        model_name=model_name,
        agents=agents,
        policy_mode=policy_mode,
    )

    output_path = output_dir / "open_harness.yaml"
    console.print(f"\n[bold]Step 7:[/bold] Confirm")
    console.print(f"Config will be written to: [bold]{output_path}[/bold]\n")
    console.print(Panel(yaml_content, title="open_harness.yaml", border_style="dim"))

    if not click.confirm("Write config?", default=True):
        console.print("[yellow]Aborted. No file written.[/yellow]")
        raise SystemExit(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml_content)
    console.print(f"[green]Config written to {output_path}[/green]")
    return output_path
