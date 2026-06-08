"""
ReportGenerator — generates HTML, JUnit XML, and JSON test reports.

Supports embedding screenshots, healing audit trails, flakiness scores,
and step-by-step execution details.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from engine.models import StepResult, StepStatus, TestResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates multi-format test execution reports."""

    async def generate(
        self, result: TestResult, fmt: str, output_path: str,
    ) -> str:
        """Generate a report in the specified format. Returns the output path."""
        match fmt:
            case "html":
                return self._generate_html(result, output_path)
            case "junit":
                return self._generate_junit_xml(result, output_path)
            case "json":
                return self._generate_json(result, output_path)
            case _:
                logger.warning("Unknown report format: %s", fmt)
                return ""

    def _generate_junit_xml(self, result: TestResult, path: str) -> str:
        """JUnit XML compatible with Jenkins, GitHub Actions, GitLab CI."""
        passed = sum(1 for s in result.steps if s.status in (StepStatus.PASSED, StepStatus.HEALED))
        failed = sum(1 for s in result.steps if s.status == StepStatus.FAILED)
        healed = sum(1 for s in result.steps if s.status == StepStatus.HEALED)

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuite name="{_xml_escape(result.test_name)}" '
            f'tests="{len(result.steps)}" failures="{failed}" '
            f'time="{result.total_duration_ms / 1000:.2f}" '
            f'timestamp="{result.started_at}">',
        ]

        for sr in result.steps:
            action = sr.action_type or "unknown"
            name = f"Step {sr.step_id}: {action}"
            time_s = sr.duration_ms / 1000

            if sr.status == StepStatus.FAILED:
                lines.append(f'  <testcase name="{_xml_escape(name)}" time="{time_s:.2f}">')
                lines.append(f'    <failure message="{_xml_escape(sr.error or "Step failed")}"/>')
                if sr.screenshot:
                    lines.append(f'    <system-out>Screenshot: {sr.screenshot}</system-out>')
                lines.append('  </testcase>')
            elif sr.status == StepStatus.HEALED:
                lines.append(f'  <testcase name="{_xml_escape(name)}" time="{time_s:.2f}">')
                lines.append(f'    <system-out>HEALED: {_xml_escape(sr.healing_details)}</system-out>')
                lines.append('  </testcase>')
            else:
                lines.append(f'  <testcase name="{_xml_escape(name)}" time="{time_s:.2f}"/>')

        lines.append('</testsuite>')

        xml_content = "\n".join(lines)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(xml_content, encoding="utf-8")
        logger.info("JUnit XML report saved → %s", path)
        return path

    def _generate_json(self, result: TestResult, path: str) -> str:
        """Machine-readable JSON report."""
        report = {
            "test_id": result.test_id,
            "test_name": result.test_name,
            "status": result.status.value,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "total_duration_ms": result.total_duration_ms,
            "tokens_used": result.tokens_used,
            "summary": {
                "total_steps": len(result.steps),
                "passed": sum(1 for s in result.steps if s.status == StepStatus.PASSED),
                "healed": sum(1 for s in result.steps if s.status == StepStatus.HEALED),
                "failed": sum(1 for s in result.steps if s.status == StepStatus.FAILED),
            },
            "steps": [
                {
                    "step_id": s.step_id,
                    "action_type": s.action_type,
                    "status": s.status.value,
                    "confidence": s.element_confidence,
                    "healed": s.healed,
                    "healing_details": s.healing_details,
                    "retry_count": s.retry_count,
                    "flakiness_score": s.flakiness_score,
                    "duration_ms": s.duration_ms,
                    "error": s.error,
                    "assertions": [
                        {
                            "id": a.assertion_id,
                            "type": a.assertion_type,
                            "status": a.status.value,
                            "message": a.message,
                            "api_endpoint": a.api_endpoint,
                            "diagnostic": a.diagnostic,
                        }
                        for a in s.assertions
                    ],
                }
                for s in result.steps
            ],
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8",
        )
        logger.info("JSON report saved → %s", path)
        return path

    def _generate_html(self, result: TestResult, path: str) -> str:
        """Standalone HTML report with step timeline and healing details."""
        passed = sum(1 for s in result.steps if s.status == StepStatus.PASSED)
        healed = sum(1 for s in result.steps if s.status == StepStatus.HEALED)
        failed = sum(1 for s in result.steps if s.status == StepStatus.FAILED)

        status_color = "#22c55e" if result.status == StepStatus.PASSED else "#ef4444"
        status_text = result.status.value.upper()

        step_rows = ""
        for s in result.steps:
            icon = {"passed": "✅", "healed": "🔧", "failed": "❌"}.get(s.status.value, "?")
            row_class = {"passed": "passed", "healed": "healed", "failed": "failed"}.get(s.status.value, "")
            healing_info = f'<span class="healed-badge">Healed: {_html_escape(s.healing_details)}</span>' if s.healed else ""
            error_info = f'<span class="error">{_html_escape(s.error)}</span>' if s.error else ""
            assertions_html = ""
            for a in s.assertions:
                a_icon = "✅" if a.status == StepStatus.PASSED else "❌"
                endpoint_badge = ""
                if a.api_endpoint:
                    parts = a.api_endpoint.split(" ", 1)
                    method_html = _html_escape(parts[0])
                    path_html = _html_escape(parts[1]) if len(parts) > 1 else ""
                    endpoint_badge = (
                        f'<span class="api-badge">'
                        f'<span class="api-method">{method_html}</span>'
                        f'<span class="api-path">{path_html}</span>'
                        f'</span> '
                    )
                assertions_html += (
                    f'<div class="assertion">{a_icon} {endpoint_badge}'
                    f'{_html_escape(a.assertion_type)}: {_html_escape(a.message)}</div>'
                )
                if a.diagnostic:
                    diag_json = json.dumps(a.diagnostic, indent=2, default=str)
                    assertions_html += (
                        '<details class="diagnostic"><summary>Diagnostic</summary>'
                        f'<pre>{_html_escape(diag_json)}</pre></details>'
                    )

            step_rows += f"""
            <tr class="{row_class}">
                <td>{s.step_id}</td>
                <td>{icon} {_html_escape(s.action_type)}</td>
                <td>{s.element_confidence:.2f}</td>
                <td>{s.duration_ms:.0f}ms</td>
                <td>{s.retry_count}</td>
                <td>{healing_info}{error_info}{assertions_html}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoMateQA Report — {_html_escape(result.test_name)}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 2rem; }}
  .header {{ background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }}
  .header h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
  .status {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 6px;
             background: {status_color}; color: white; font-weight: 600; }}
  .cards {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }}
  .card {{ background: #1e293b; border-radius: 8px; padding: 1rem; flex: 1; min-width: 120px; }}
  .card .value {{ font-size: 1.5rem; font-weight: 700; }}
  .card .label {{ color: #94a3b8; font-size: 0.875rem; }}
  table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; }}
  th {{ background: #334155; padding: 0.75rem; text-align: left; font-size: 0.875rem; }}
  td {{ padding: 0.75rem; border-top: 1px solid #334155; font-size: 0.875rem; }}
  tr.failed td {{ background: rgba(239,68,68,0.1); }}
  tr.healed td {{ background: rgba(234,179,8,0.1); }}
  .healed-badge {{ background: #854d0e; color: #fef3c7; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; }}
  .error {{ color: #f87171; }}
  .assertion {{ font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }}
  .api-badge {{ display: inline-flex; align-items: center; gap: 0; margin-right: 5px; border-radius: 4px; overflow: hidden; font-size: 0.72rem; font-family: monospace; vertical-align: middle; }}
  .api-method {{ background: #166534; color: #86efac; padding: 1px 5px; font-weight: 700; }}
  .api-path {{ background: #1e293b; color: #7dd3fc; padding: 1px 6px; }}
  .diagnostic {{ margin-top: 4px; }} .diagnostic summary {{ color: #64748b; font-size: 0.75rem; cursor: pointer; }}
  .footer {{ margin-top: 2rem; text-align: center; color: #475569; font-size: 0.8rem; }}
</style>
</head>
<body>
<div class="header">
  <h1>{_html_escape(result.test_name)}</h1>
  <div><span class="status">{status_text}</span>
  <span style="margin-left: 1rem; color: #94a3b8;">
    Duration: {result.total_duration_ms:.0f}ms | Started: {result.started_at}
  </span></div>
</div>

<div class="cards">
  <div class="card"><div class="value" style="color:#22c55e">{passed}</div><div class="label">Passed</div></div>
  <div class="card"><div class="value" style="color:#eab308">{healed}</div><div class="label">Healed</div></div>
  <div class="card"><div class="value" style="color:#ef4444">{failed}</div><div class="label">Failed</div></div>
  <div class="card"><div class="value">{len(result.steps)}</div><div class="label">Total Steps</div></div>
  <div class="card"><div class="value">{result.tokens_used}</div><div class="label">Tokens Used</div></div>
</div>

<table>
<thead><tr><th>#</th><th>Action</th><th>Confidence</th><th>Duration</th><th>Retries</th><th>Details</th></tr></thead>
<tbody>{step_rows}</tbody>
</table>

<div class="footer">Generated by AutoMateQA | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</div>
</body></html>"""

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(html, encoding="utf-8")
        logger.info("HTML report saved → %s", path)
        return path


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
