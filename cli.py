"""
CLI entrypoint for the AutoMateQA Self-Healing Automation Engine.

Commands:
  record      – Launch browser, record user actions + assertions, save test model.
  execute     – Load a saved test model and replay it (with optional healing).
  inspect     – Pretty-print a saved test model.
  drift-check – Check selectors for drift without executing actions.
  suite       – Execute multiple tests in batch (sequential or parallel).
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
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from engine.core import TestEngine
from engine.models import HealingMode, StepStatus, TestSuite

console = Console()


def _read_test_json(path: str) -> dict:
    """Read a test file (.json or .aqa) and return parsed dict."""
    p = Path(path)
    if p.suffix == ".aqa":
        import gzip
        with gzip.open(p, "rb") as f:
            return json.loads(f.read().decode("utf-8"))
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve_test_path(name: str) -> str:
    """Resolve a test name/path to a full file path.

    - 'my_test'       → './my_test.aqa'  (default .aqa extension)
    - 'my_test.json'  → './my_test.json' (explicit extension preserved)
    - 'my_test.aqa'   → './my_test.aqa'  (explicit extension preserved)
    - 'dir/test.json' → 'dir/test.json'  (paths with dirs preserved as-is)
    """
    p = Path(name)
    if p.suffix in (".json", ".aqa"):
        return str(p)
    return str(p.with_suffix(".aqa"))


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
@click.version_option(version="0.2.0", prog_name="automateqa")
def cli():
    """AutoMateQA – AI-powered self-healing test automation engine."""
    pass


# ------------------------------------------------------------------
# RECORD command
# ------------------------------------------------------------------


@cli.command()
@click.option("--url", required=True, help="Starting URL for the recording session.")
@click.option("--output", "-o", default="test", help="Test name (saved as <name>.aqa in current directory).")
@click.option("--name", default="Recorded Test", help="Name of the test.")
@click.option("--har", "record_har", is_flag=True, default=False, help="Capture HAR file of network activity.")
@click.option("--har-path", default="trace.har", help="Output path for the HAR file.")
@click.option("--trace", "record_trace", is_flag=True, default=False, help="Record a Playwright trace.")
@click.option("--trace-path", default="trace.zip", help="Output path for the trace zip.")
@click.option("--video", "record_video", is_flag=True, default=False, help="Record session video.")
@click.option("--no-enrich", "skip_enrich", is_flag=True, default=False, help="Skip Playwright locator enrichment.")
@click.option("--capture-network", is_flag=True, default=False, help="Capture per-step API calls for targeted waits.")
@click.option("--browser", "browser_type", default="chromium", type=click.Choice(["chromium", "firefox", "webkit"]))
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def record(url, output, name, record_har, har_path, record_trace, trace_path,
           record_video, skip_enrich, capture_network, browser_type, verbose):
    """Launch a browser and record user interactions."""
    _setup_logging(verbose)

    output = _resolve_test_path(output)

    console.print(Panel(
        f"[bold cyan]Recording[/bold cyan] → {url}\nOutput: [green]{output}[/green]",
        title="Recording Mode",border_style="cyan",
    ))

    engine = TestEngine(
        llm_enabled=False,
        healing_mode="disabled",
        headless=False,
        verbose=verbose,
        browser_type=browser_type,
        record_har=record_har,
        har_path=har_path,
        record_trace=record_trace,
        trace_path=trace_path,
        record_video=record_video,
        enrich_with_playwright_locators=not skip_enrich,
        capture_network=capture_network,
    )

    try:
        asyncio.run(engine.record(url=url, save_path=output, test_name=name))
    except KeyboardInterrupt:
        console.print("\n[yellow]Recording interrupted.[/yellow]")

    console.print(f"[bold green]Test saved to {output}[/bold green]")


# ------------------------------------------------------------------
# EXECUTE command
# ------------------------------------------------------------------


@cli.command()
@click.argument("test_file", type=click.Path(exists=True))
@click.option("--llm/--no-llm", default=False, help="Enable LLM-backed healing.")
@click.option("--healing-mode", type=click.Choice(["disabled", "strict", "auto_update", "debug"]), default="disabled")
@click.option("--confidence", type=float, default=0.75, help="Confidence threshold.")
@click.option("--model", "llm_model", default="gpt-4o", help="LLM model for healing.")
@click.option("--provider", "llm_provider", default="openai", type=click.Choice(["openai", "anthropic", "local"]))
@click.option("--headless", is_flag=True, help="Run headless.")
@click.option("--browser", "browser_type", default="chromium", type=click.Choice(["chromium", "firefox", "webkit"]))
@click.option("--device", "device_name", default="", help='Device emulation (e.g. "iPhone 14").')
@click.option("--locale", default="", help='Browser locale (e.g. "en-US").')
@click.option("--storage-state", "storage_state_path", default="", help="Auth state JSON path.")
@click.option("--save-storage-state", is_flag=True, help="Save auth state after execution.")
@click.option("--screenshot-dir", default="screenshots", help="Directory for failure screenshots.")
@click.option("--trace", "record_trace", is_flag=True, default=False, help="Record Playwright trace.")
@click.option("--trace-path", default="trace.zip", help="Trace output path.")
@click.option("--video", "record_video", is_flag=True, default=False, help="Record session video.")
@click.option("--retry-strategy", default="exponential", type=click.Choice(["none", "linear", "exponential"]))
@click.option("--max-retries", "max_step_retries", default=3, type=int, help="Max retries per step.")
@click.option("--report", "report_format", default="", type=click.Choice(["", "html", "junit", "json"]), help="Report format.")
@click.option("--report-path", default="", help="Report output path.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def execute(test_file, llm, healing_mode, confidence, llm_model, llm_provider,
            headless, browser_type, device_name, locale, storage_state_path,
            save_storage_state, screenshot_dir, record_trace, trace_path,
            record_video, retry_strategy, max_step_retries,
            report_format, report_path, verbose):
    """Execute a recorded test with optional self-healing."""
    _setup_logging(verbose)

    healing = healing_mode if llm else "disabled"

    console.print(Panel(
        f"[bold cyan]Executing[/bold cyan] → {test_file}\n"
        f"LLM: [{'green' if llm else 'red'}]{llm}[/{'green' if llm else 'red'}]  "
        f"Healing: [yellow]{healing}[/yellow]  "
        f"Confidence: {confidence}  Retries: {max_step_retries}",
        title="Execute Mode", border_style="cyan",
    ))

    engine = TestEngine(
        llm_enabled=llm,
        healing_mode=healing,
        confidence_threshold=confidence,
        headless=headless,
        verbose=verbose,
        llm_model=llm_model,
        llm_provider=llm_provider,
        browser_type=browser_type,
        device_name=device_name,
        locale=locale,
        storage_state_path=storage_state_path,
        save_storage_state=save_storage_state,
        record_trace=record_trace,
        trace_path=trace_path,
        record_video=record_video,
        retry_strategy=retry_strategy,
        max_step_retries=max_step_retries,
        report_format=report_format,
        report_path=report_path,
    )

    test_model_data = _read_test_json(test_file)

    result = asyncio.run(engine.execute(test_path=test_file, screenshot_dir=screenshot_dir))

    _display_results(result, test_model_data, verbose)

    if result.status == StepStatus.FAILED:
        sys.exit(1)


# ------------------------------------------------------------------
# DRIFT-CHECK command (Phase 20)
# ------------------------------------------------------------------


@cli.command("drift-check")
@click.argument("test_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--threshold", type=float, default=0.6, help="Confidence threshold for drift alert.")
@click.option("--browser", "browser_type", default="chromium", type=click.Choice(["chromium", "firefox", "webkit"]))
@click.option("--headless/--no-headless", default=True, help="Run headless.")
@click.option("--verbose", "-v", is_flag=True)
def drift_check(test_files, threshold, browser_type, headless, verbose):
    """Check selectors for drift without executing actions."""
    _setup_logging(verbose)

    engine = TestEngine(
        llm_enabled=False, healing_mode="disabled",
        headless=headless, browser_type=browser_type, verbose=verbose,
    )

    has_drift = False
    for test_file in test_files:
        console.print(f"\n[cyan]Checking[/cyan] {test_file}")
        alerts = asyncio.run(engine.drift_check(test_file, threshold=threshold))

        if alerts:
            has_drift = True
            for alert in alerts:
                icon = "[red]!!![/red]" if alert["severity"] == "critical" else "[yellow]!![/yellow]"
                console.print(
                    f"  {icon} Step {alert['step_id']} ({alert['action']}): "
                    f"confidence {alert['confidence']:.2f} — {alert['selector'][:60]}"
                )
        else:
            console.print(f"  [green]All selectors stable (confidence >= {threshold})[/green]")

    if has_drift:
        sys.exit(1)


# ------------------------------------------------------------------
# SUITE command (Phase 18)
# ------------------------------------------------------------------


@cli.command()
@click.argument("test_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--parallel", is_flag=True, help="Run tests in parallel.")
@click.option("--workers", default=4, type=int, help="Max parallel workers.")
@click.option("--stop-on-failure", is_flag=True, help="Stop on first failure (sequential only).")
@click.option("--headless/--no-headless", default=True)
@click.option("--llm/--no-llm", default=False)
@click.option("--healing-mode", default="disabled", type=click.Choice(["disabled", "strict", "auto_update"]))
@click.option("--report", "report_format", default="", type=click.Choice(["", "html", "junit", "json"]))
@click.option("--report-path", default="")
@click.option("--verbose", "-v", is_flag=True)
def suite(test_files, parallel, workers, stop_on_failure, headless, llm,
          healing_mode, report_format, report_path, verbose):
    """Execute multiple tests as a suite."""
    _setup_logging(verbose)

    healing = healing_mode if llm else "disabled"

    engine = TestEngine(
        llm_enabled=llm, healing_mode=healing, headless=headless,
        verbose=verbose, report_format=report_format, report_path=report_path,
    )

    test_suite = TestSuite(
        name="CLI Suite",
        test_paths=list(test_files),
        parallel=parallel,
        max_workers=workers,
        stop_on_first_failure=stop_on_failure,
    )

    results = asyncio.run(engine.execute_suite(test_suite))

    # Summary
    passed = sum(1 for r in results if r.status == StepStatus.PASSED)
    failed = sum(1 for r in results if r.status == StepStatus.FAILED)
    console.print(f"\n[bold]Suite Results: {passed}/{len(results)} passed, {failed} failed[/bold]")

    if failed > 0:
        sys.exit(1)


# ------------------------------------------------------------------
# INSPECT command
# ------------------------------------------------------------------


@cli.command()
@click.argument("test_file", type=click.Path(exists=True))
def inspect(test_file):
    """Pretty-print a saved test model."""
    _setup_logging(False)

    data = _read_test_json(test_file)

    console.print(Panel(
        f"[bold]{data.get('name', 'Unknown')}[/bold]\n"
        f"ID: {data.get('test_id', '—')}\n"
        f"URL: [cyan]{data.get('base_url', '—')}[/cyan]\n"
        f"Steps: {len(data.get('steps', []))}\n"
        f"Created: {data.get('created_at', '—')}",
        title="Test Inspection", border_style="cyan",
    ))

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

    config = data.get("config", {})
    console.print(Panel(
        Syntax(json.dumps(config, indent=2), "json", theme="monokai"),
        title="Config", border_style="dim",
    ))


# ------------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------------


def _display_results(result, test_model_data: dict, verbose: bool) -> None:
    """Assertion-focused output with optional verbose step details."""
    steps_data = test_model_data.get("steps", [])
    failed_assertions: list[tuple[int, str, dict, object]] = []

    console.print()
    for i, step_result in enumerate(result.steps):
        step_data = steps_data[i] if i < len(steps_data) else {}
        action_type = step_data.get("action", {}).get("action_type", "?")

        icon = (
            "[green]PASS[/green]"
            if step_result.status == StepStatus.PASSED
            else "[yellow]HEALED[/yellow]"
            if step_result.status == StepStatus.HEALED
            else "[red]FAIL[/red]"
        )

        extra = ""
        if verbose:
            extra = (
                f"  [dim](conf={step_result.element_confidence:.2f}, "
                f"{step_result.duration_ms:.0f}ms"
                f"{f', retries={step_result.retry_count}' if step_result.retry_count else ''}"
                f"{', healed' if step_result.healed else ''})[/dim]"
            )
        console.print(f"  {icon} Step {step_result.step_id} ({action_type}){extra}")

        assertion_models = step_data.get("assertions", [])
        for j, ar in enumerate(step_result.assertions):
            a_data = assertion_models[j] if j < len(assertion_models) else {}
            fp = a_data.get("fingerprint", {})
            tag = fp.get("tag_name", "?")
            text = fp.get("text_content", "")[:40]
            element_desc = f'{tag} "{text}"' if text else tag

            if ar.status == StepStatus.PASSED:
                console.print(f"       [green]PASS[/green] {ar.assertion_type}: {element_desc}")
            else:
                console.print(
                    f"       [red]FAIL[/red] {ar.assertion_type}: {element_desc} "
                    f"[dim]— {ar.message}[/dim]"
                )
                failed_assertions.append((step_result.step_id, ar.assertion_type, fp, ar))

        if step_result.status == StepStatus.FAILED and not step_result.assertions and step_result.error:
            console.print(f"       [red]{step_result.error}[/red]")

    if failed_assertions:
        console.print()
        console.rule("[bold red]Failure Summary[/bold red]")
        for step_id, a_type, fp, ar in failed_assertions:
            tag = fp.get("tag_name", "?")
            text = fp.get("text_content", "")[:50]
            console.print(
                f"  Step {step_id} → [bold]{a_type}[/bold] ({tag} \"{text}\"): [red]{ar.message}[/red]"
            )

    passed = sum(1 for s in result.steps if s.status in (StepStatus.PASSED, StepStatus.HEALED))
    total = len(result.steps)
    overall_map = {
        StepStatus.PASSED: "[bold green]ALL PASSED[/bold green]",
        StepStatus.HEALED: "[bold yellow]PASSED (with healing)[/bold yellow]",
        StepStatus.FAILED: "[bold red]FAILED[/bold red]",
    }
    overall = overall_map.get(result.status, result.status.value)

    console.print(
        f"\nOverall: {overall}  |  Duration: {result.total_duration_ms:.0f}ms  |  "
        f"Steps: {passed}/{total}  |  Healed: {result.healed_count}  |  "
        f"Tokens: {result.tokens_used}"
    )

    if verbose:
        console.print()
        table = Table(title="Detailed Step Results", show_lines=True)
        table.add_column("Step", justify="center", style="bold")
        table.add_column("Action", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Confidence", justify="center")
        table.add_column("Retries", justify="center")
        table.add_column("Flakiness", justify="center")
        table.add_column("Duration", justify="right")
        table.add_column("Error", style="red", max_width=40)

        for step in result.steps:
            status_style = {
                StepStatus.PASSED: "[green]PASSED[/green]",
                StepStatus.HEALED: "[yellow]HEALED[/yellow]",
                StepStatus.FAILED: "[red]FAILED[/red]",
            }.get(step.status, step.status.value)
            table.add_row(
                str(step.step_id),
                step.action_type or "—",
                status_style,
                f"{step.element_confidence:.2f}",
                str(step.retry_count),
                f"{step.flakiness_score:.2f}" if step.flakiness_score > 0 else "—",
                f"{step.duration_ms:.0f}ms",
                step.error[:40] or "—",
            )
        console.print(table)


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    cli()
