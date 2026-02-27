"""
CLI entrypoint for the Self-Healing Automation Engine.

Commands:
  record   – Launch browser, record user actions + assertions, save test model.
  execute  – Load a saved test model and replay it (with optional healing).
  inspect  – Pretty-print a saved test model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from engine.core import TestEngine
from engine.models import HealingMode, StepStatus

console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


# ------------------------------------------------------------------
# CLI group
# ------------------------------------------------------------------


@click.group()
@click.version_option(version="0.1.0", prog_name="testing-automation-agent")
def cli():
    """Self-Healing Automation Engine – record, execute, and inspect tests."""
    pass


# ------------------------------------------------------------------
# RECORD command
# ------------------------------------------------------------------


@cli.command()
@click.option("--url", required=True, help="Starting URL for the recording session.")
@click.option(
    "--output", "-o", default="test.json", help="Output path for the test model JSON."
)
@click.option("--name", default="Recorded Test", help="Name of the test.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def record(url: str, output: str, name: str, verbose: bool):
    """Launch a browser and record user interactions."""
    _setup_logging(verbose)

    console.print(
        Panel(
            f"[bold cyan]Recording[/bold cyan] → {url}\n"
            f"Output: [green]{output}[/green]",
            title="🎬 Record Mode",
            border_style="cyan",
        )
    )

    engine = TestEngine(
        llm_enabled=False,
        healing_mode="disabled",
        headless=False,
        verbose=verbose,
    )

    try:
        asyncio.run(engine.record(url=url, save_path=output, test_name=name))
    except KeyboardInterrupt:
        console.print("\n[yellow]Recording interrupted.[/yellow]")

    console.print(f"[bold green]✅ Test saved to {output}[/bold green]")


# ------------------------------------------------------------------
# EXECUTE command
# ------------------------------------------------------------------


@cli.command()
@click.argument("test_file", type=click.Path(exists=True))
@click.option("--llm/--no-llm", default=False, help="Enable LLM-backed healing.")
@click.option(
    "--healing-mode",
    type=click.Choice(["disabled", "strict", "auto_update", "debug"]),
    default="disabled",
    help="Healing mode.",
)
@click.option(
    "--confidence",
    type=float,
    default=0.75,
    help="Confidence threshold for selectors.",
)
@click.option(
    "--model",
    "llm_model",
    default="gpt-4o",
    help="OpenAI model to use for healing.",
)
@click.option("--headless", is_flag=True, help="Run browser in headless mode.")
@click.option(
    "--screenshot-dir",
    default="screenshots",
    help="Directory for failure screenshots.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def execute(
    test_file: str,
    llm: bool,
    healing_mode: str,
    confidence: float,
    llm_model: str,
    headless: bool,
    screenshot_dir: str,
    verbose: bool,
):
    """Execute a recorded test with optional self-healing."""
    _setup_logging(verbose)

    healing = healing_mode if llm else "disabled"

    console.print(
        Panel(
            f"[bold cyan]Executing[/bold cyan] → {test_file}\n"
            f"LLM: [{'green' if llm else 'red'}]{llm}[/{'green' if llm else 'red'}]  "
            f"Healing: [yellow]{healing}[/yellow]  "
            f"Confidence: {confidence}",
            title="▶ Execute Mode",
            border_style="cyan",
        )
    )

    engine = TestEngine(
        llm_enabled=llm,
        healing_mode=healing,
        confidence_threshold=confidence,
        headless=headless,
        verbose=verbose,
        llm_model=llm_model,
    )

    # Load test model for assertion display context
    test_model_data = json.loads(Path(test_file).read_text(encoding="utf-8"))

    result = asyncio.run(
        engine.execute(test_path=test_file, screenshot_dir=screenshot_dir)
    )

    _display_results(result, test_model_data, verbose)

    if result.status == StepStatus.FAILED:
        sys.exit(1)


def _display_results(result, test_model_data: dict, verbose: bool) -> None:
    """Assertion-focused output with optional verbose step details."""
    steps_data = test_model_data.get("steps", [])
    failed_assertions: list[tuple[int, str, dict, object]] = []

    console.print()
    for i, step_result in enumerate(result.steps):
        step_data = steps_data[i] if i < len(steps_data) else {}
        action_type = step_data.get("action", {}).get("action_type", "?")

        icon = (
            "[green]✅[/green]"
            if step_result.status == StepStatus.PASSED
            else "[yellow]🔧[/yellow]"
            if step_result.status == StepStatus.HEALED
            else "[red]❌[/red]"
        )

        # Step header
        extra = ""
        if verbose:
            extra = (
                f"  [dim](confidence={step_result.element_confidence:.2f}, "
                f"{step_result.duration_ms:.0f}ms"
                f"{', healed' if step_result.healed else ''})[/dim]"
            )
        console.print(f"  {icon} Step {step_result.step_id} ({action_type}){extra}")

        # Assertion details
        assertion_models = step_data.get("assertions", [])
        for j, ar in enumerate(step_result.assertions):
            a_data = assertion_models[j] if j < len(assertion_models) else {}
            fp = a_data.get("fingerprint", {})
            tag = fp.get("tag_name", "?")
            text = fp.get("text_content", "")[:40]
            element_desc = f'{tag} "{text}"' if text else tag

            if ar.status == StepStatus.PASSED:
                console.print(f"       [green]✅[/green] {ar.assertion_type}: {element_desc}")
            else:
                console.print(
                    f"       [red]❌[/red] {ar.assertion_type}: {element_desc} "
                    f"[dim]— {ar.message}[/dim]"
                )
                failed_assertions.append(
                    (step_result.step_id, ar.assertion_type, fp, ar)
                )

        # Show step-level error only if no assertions and step failed
        if (
            step_result.status == StepStatus.FAILED
            and not step_result.assertions
            and step_result.error
        ):
            console.print(f"       [red]{step_result.error}[/red]")

    # Failure summary
    if failed_assertions:
        console.print()
        console.rule("[bold red]Failure Summary[/bold red]")
        for step_id, a_type, fp, ar in failed_assertions:
            tag = fp.get("tag_name", "?")
            text = fp.get("text_content", "")[:50]
            selector = fp.get("css_selector", "")
            console.print(
                f"  Step {step_id} → [bold]{a_type}[/bold] "
                f"({tag} \"{text}\"): [red]{ar.message}[/red]"
            )
            if verbose and selector:
                console.print(f"           [dim]selector: {selector}[/dim]")

    # Overall
    passed = sum(
        1 for s in result.steps
        if s.status in (StepStatus.PASSED, StepStatus.HEALED)
    )
    total = len(result.steps)
    overall_map = {
        StepStatus.PASSED: "[bold green]✅ ALL PASSED[/bold green]",
        StepStatus.HEALED: "[bold yellow]🔧 PASSED (with healing)[/bold yellow]",
        StepStatus.FAILED: "[bold red]❌ FAILED[/bold red]",
    }
    overall = overall_map.get(result.status, result.status.value)

    console.print(
        f"\nOverall: {overall}  |  "
        f"Duration: {result.total_duration_ms:.0f}ms  |  "
        f"Steps: {passed}/{total} passed"
    )

    # Verbose: detailed step table
    if verbose:
        console.print()
        table = Table(title="Detailed Step Results", show_lines=True)
        table.add_column("Step", justify="center", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Confidence", justify="center")
        table.add_column("Healed", justify="center")
        table.add_column("Duration (ms)", justify="right")
        table.add_column("Error", style="red")

        for step in result.steps:
            status_style = {
                StepStatus.PASSED: "[green]PASSED[/green]",
                StepStatus.HEALED: "[yellow]HEALED[/yellow]",
                StepStatus.FAILED: "[red]FAILED[/red]",
            }.get(step.status, step.status.value)
            table.add_row(
                str(step.step_id),
                status_style,
                f"{step.element_confidence:.2f}",
                "🔧" if step.healed else "—",
                f"{step.duration_ms:.1f}",
                step.error or "—",
            )
        console.print(table)


# ------------------------------------------------------------------
# INSPECT command
# ------------------------------------------------------------------


@cli.command()
@click.argument("test_file", type=click.Path(exists=True))
def inspect(test_file: str):
    """Pretty-print a saved test model."""
    _setup_logging(False)

    path = Path(test_file)
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)

    console.print(
        Panel(
            f"[bold]{data.get('name', 'Unknown')}[/bold]\n"
            f"ID: {data.get('test_id', '—')}\n"
            f"URL: [cyan]{data.get('base_url', '—')}[/cyan]\n"
            f"Steps: {len(data.get('steps', []))}\n"
            f"Created: {data.get('created_at', '—')}",
            title="🔍 Test Inspection",
            border_style="cyan",
        )
    )

    # Steps summary
    table = Table(title="Steps", show_lines=True)
    table.add_column("#", justify="center", style="bold")
    table.add_column("Action", style="cyan")
    table.add_column("Target", style="green", max_width=50)
    table.add_column("Assertions", justify="center")

    for step in data.get("steps", []):
        action = step.get("action", {})
        target = step.get("target", {})
        action_type = action.get("action_type", "—")
        selector = target.get("css_selector", "") or target.get("data_testid", "") or "—"
        if action_type == "navigate":
            selector = action.get("url", "")[:50]

        table.add_row(
            str(step.get("step_id", "—")),
            action_type,
            selector,
            str(len(step.get("assertions", []))),
        )

    console.print(table)

    # Config
    config = data.get("config", {})
    console.print(
        Panel(
            Syntax(json.dumps(config, indent=2), "json", theme="monokai"),
            title="⚙️ Config",
            border_style="dim",
        )
    )


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    cli()
