"""Reporter should include network diagnostic for failed api_call assertions."""

import json
import tempfile
import unittest
from pathlib import Path

from engine.models import AssertionResult, StepResult, StepStatus, TestResult
from engine.reporter import ReportGenerator

from tests.api_assertion.conftest import run


class TestReporterBlock(unittest.TestCase):
    def _build_result(self) -> TestResult:
        return TestResult(
            test_id="t1",
            test_name="demo",
            started_at="2026-04-17T00:00:00+00:00",
            finished_at="2026-04-17T00:00:01+00:00",
            status=StepStatus.FAILED,
            steps=[
                StepResult(
                    step_id=1,
                    status=StepStatus.FAILED,
                    action_type="click",
                    assertions=[
                        AssertionResult(
                            assertion_id="a1",
                            assertion_type="api_call",
                            status=StepStatus.FAILED,
                            message="mismatch",
                            diagnostic={
                                "expected": {
                                    "method": "POST",
                                    "path_template": "/x",
                                },
                                "observed": {
                                    "matched_call": None,
                                    "step_network_window": [],
                                },
                            },
                        )
                    ],
                )
            ],
        )

    def test_json_report_contains_diagnostic(self):
        result = self._build_result()
        rep = ReportGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "report.json")
            run(rep.generate(result, "json", out_path))
            payload = json.loads(Path(out_path).read_text(encoding="utf-8"))

        steps = payload["steps"]
        self.assertEqual(
            steps[0]["assertions"][0]["diagnostic"]["expected"]["path_template"],
            "/x",
        )
        self.assertEqual(
            steps[0]["assertions"][0]["diagnostic"]["expected"]["method"],
            "POST",
        )
        self.assertIsNone(
            steps[0]["assertions"][0]["diagnostic"]["observed"]["matched_call"]
        )

    def test_html_report_contains_diagnostic_details_block(self):
        result = self._build_result()
        rep = ReportGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "report.html")
            run(rep.generate(result, "html", out_path))
            html = Path(out_path).read_text(encoding="utf-8")

        self.assertIn("<details", html)
        self.assertIn("Diagnostic", html)
        self.assertIn("path_template", html)


if __name__ == "__main__":
    unittest.main()
