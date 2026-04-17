"""
ApiAssertionEvaluator — evaluates API_CALL assertions against live responses.

Flow per assertion:
  1. Race page.wait_for_response(predicate) against page.wait_for_load_state("networkidle").
  2. If a matching response wins, evaluate the chosen target.
  3. If networkidle or timeout wins, fail with a structured diagnostic
     listing every XHR/fetch response observed in the current step window.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Optional
from urllib.parse import urlparse, parse_qs

from playwright.async_api import Page, Response

from engine.models import (
    ApiAssertionOp,
    ApiAssertionSpec,
    ApiAssertionTarget,
    Assertion,
    AssertionResult,
    EngineConfig,
    StepStatus,
)
from engine.path_template import path_matches_template

logger = logging.getLogger(__name__)


class ApiAssertionEvaluator:
    """Evaluates assertions of type API_CALL against a Playwright page."""

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._step_window: list[dict[str, Any]] = []
        self._step_listener = None
        self._step_page: Optional[Page] = None

    # ------------------------------------------------------------------
    # Step window (diagnostic capture)
    # ------------------------------------------------------------------

    def start_step_window(self, page: Page) -> None:
        """Begin recording responses for the current step (for failure diagnostics)."""
        self._step_window = []
        self._step_page = page

        async def _on_response(resp: Response) -> None:
            try:
                if resp.request.resource_type not in ("xhr", "fetch"):
                    return
                timing = getattr(resp.request, "timing", None) or {}
                end = float(timing.get("responseEnd") or 0.0)
                start = float(timing.get("startTime") or 0.0)
                self._step_window.append({
                    "method": resp.request.method,
                    "url": resp.url,
                    "status": resp.status,
                    "duration_ms": max(0, int(end - start)),
                })
            except Exception:
                pass

        def _sync(resp: Response) -> None:
            asyncio.ensure_future(_on_response(resp))

        self._step_listener = _sync
        page.on("response", _sync)

    def end_step_window(self) -> None:
        """Stop recording responses for the current step."""
        if self._step_page and self._step_listener:
            try:
                self._step_page.remove_listener("response", self._step_listener)
            except Exception:
                pass
        self._step_listener = None
        self._step_page = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def evaluate(self, page: Page, assertion: Assertion) -> AssertionResult:
        spec = assertion.api_spec
        result = AssertionResult(
            assertion_id=assertion.assertion_id,
            assertion_type=assertion.assertion_type.value,
        )

        if spec is None:
            result.status = StepStatus.FAILED
            result.message = "API_CALL assertion missing api_spec"
            return result

        matched = await self._await_match(page, spec)
        if matched is None:
            result.status = StepStatus.FAILED
            result.message = self._build_match_failure_message(spec)
            result.diagnostic = self._build_diagnostic(spec, matched=None)
            return result

        await self._dispatch_target(matched, spec, result)

        if result.status == StepStatus.FAILED:
            result.diagnostic = self._build_diagnostic(spec, matched=matched)
        return result

    # ------------------------------------------------------------------
    # Response matching race
    # ------------------------------------------------------------------

    async def _await_match(
        self, page: Page, spec: ApiAssertionSpec
    ) -> Optional[Response]:
        predicate = self._make_predicate(spec)
        match_task = asyncio.create_task(page.wait_for_response(predicate))
        idle_task = asyncio.create_task(page.wait_for_load_state("networkidle"))
        timeout_s = max(0.1, self._config.api_assertion_match_timeout_ms / 1000.0)

        done, pending = await asyncio.wait(
            {match_task, idle_task},
            return_when=asyncio.FIRST_COMPLETED,
            timeout=timeout_s,
        )

        for t in pending:
            t.cancel()

        if match_task in done and not match_task.cancelled():
            exc = match_task.exception()
            if exc is None:
                return match_task.result()
        return None

    @staticmethod
    def _make_predicate(spec: ApiAssertionSpec):
        def _pred(response: Response) -> bool:
            try:
                if response.request.method.upper() != spec.method.upper():
                    return False
                u = urlparse(response.url)
                if not path_matches_template(u.path, spec.path_template):
                    return False
                if spec.query_keys_present:
                    qs = parse_qs(u.query, keep_blank_values=True)
                    if not all(k in qs for k in spec.query_keys_present):
                        return False
                return True
            except Exception:
                return False
        return _pred

    def _build_match_failure_message(self, spec: ApiAssertionSpec) -> str:
        seen = ", ".join(
            f"{c['method']} {c['url']} [{c['status']}]"
            for c in self._step_window
        ) or "(none)"
        return (
            f"Expected {spec.method} {spec.path_template}; "
            f"not seen before networkidle. Observed: {seen}"
        )

    def _build_diagnostic(
        self, spec: ApiAssertionSpec, matched: Optional[Response]
    ) -> dict[str, Any]:
        expected = {
            "method": spec.method,
            "path_template": spec.path_template,
            "target": spec.target.value,
            "op": spec.op.value,
            "jsonpath": spec.jsonpath,
            "header_name": spec.header_name,
            "expected_value": spec.expected,
        }
        observed_match: Optional[dict[str, Any]] = None
        if matched is not None:
            observed_match = {
                "method": matched.request.method,
                "url": matched.url,
                "status": matched.status,
            }
        return {
            "expected": expected,
            "observed": {
                "matched_call": observed_match,
                "step_network_window": list(self._step_window),
            },
        }

    # ------------------------------------------------------------------
    # Target dispatch
    # ------------------------------------------------------------------

    async def _dispatch_target(
        self, response: Response, spec: ApiAssertionSpec, result: AssertionResult
    ) -> None:
        try:
            match spec.target:
                case ApiAssertionTarget.STATUS:
                    self._check_status(response, spec, result)
                case ApiAssertionTarget.RESPONSE_JSONPATH:
                    body = await self._response_json(response)
                    self._check_json(body, spec, result, side="response")
                case ApiAssertionTarget.REQUEST_JSONPATH:
                    body = self._request_json(response)
                    self._check_json(body, spec, result, side="request")
                case ApiAssertionTarget.RESPONSE_HEADER:
                    self._check_header(response.headers, spec, result, "response")
                case ApiAssertionTarget.REQUEST_HEADER:
                    self._check_header(response.request.headers, spec, result, "request")
                case ApiAssertionTarget.RESPONSE_SCHEMA:
                    await self._check_schema(response, spec, result)
                case ApiAssertionTarget.RESPONSE_TIME_MS:
                    self._check_time(response, spec, result)
                case _:
                    result.status = StepStatus.FAILED
                    result.message = f"Unsupported target: {spec.target.value}"
        except ValueError as ve:
            result.status = StepStatus.FAILED
            result.message = str(ve)
        except Exception as e:
            result.status = StepStatus.FAILED
            result.message = f"Evaluator error: {e}"

    # ------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------

    @staticmethod
    def _check_status(
        response: Response, spec: ApiAssertionSpec, result: AssertionResult
    ) -> None:
        actual = str(response.status)
        expected = spec.expected.strip().lower()

        if expected.endswith("xx") and len(expected) == 3 and expected[0].isdigit():
            class_digit = expected[0]
            matches = actual.startswith(class_digit)
            passed = matches if spec.op is ApiAssertionOp.EQUALS else not matches
            result.status = StepStatus.PASSED if passed else StepStatus.FAILED
            result.message = (
                f"Status {actual} {'matches' if passed else 'does not match'} {expected}"
            )
            return

        passed = _compare(actual, spec.expected, spec.op)
        result.status = StepStatus.PASSED if passed else StepStatus.FAILED
        result.message = (
            f"Status expected {spec.op.value} {spec.expected!r}; got {actual}"
        )

    # ------------------------------------------------------------------
    # JSONPath
    # ------------------------------------------------------------------

    @staticmethod
    async def _response_json(response: Response) -> Any:
        try:
            return await response.json()
        except Exception as e:
            raise ValueError(f"response not JSON: {e}")

    @staticmethod
    def _request_json(response: Response) -> Any:
        data = response.request.post_data_json
        if data is None:
            raise ValueError("no request body")
        return data

    @staticmethod
    def _check_json(
        body_source: Any,
        spec: ApiAssertionSpec,
        result: AssertionResult,
        side: str,
    ) -> None:
        from jsonpath_ng.ext import parse as jp_parse

        if not spec.jsonpath:
            result.status = StepStatus.FAILED
            result.message = f"{side}_jsonpath: jsonpath is empty"
            return

        try:
            expr = jp_parse(spec.jsonpath)
        except Exception as e:
            result.status = StepStatus.FAILED
            result.message = f"{side}_jsonpath: invalid expression: {e}"
            return

        matches = [m.value for m in expr.find(body_source)]
        if not matches:
            if spec.op is ApiAssertionOp.NOT_EXISTS:
                result.status = StepStatus.PASSED
                result.message = f"{side}_jsonpath: path not found (op=not_exists)"
                return
            result.status = StepStatus.FAILED
            result.message = f"{side}_jsonpath: path '{spec.jsonpath}' not found in body"
            return

        actual = matches[0] if len(matches) == 1 else matches
        passed = _compare(actual, spec.expected, spec.op)
        result.status = StepStatus.PASSED if passed else StepStatus.FAILED
        result.message = (
            f"{side}_jsonpath {spec.jsonpath} {spec.op.value} "
            f"{spec.expected!r}; got {actual!r}"
        )

    # ------------------------------------------------------------------
    # Headers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_header(
        headers: dict, spec: ApiAssertionSpec, result: AssertionResult, side: str
    ) -> None:
        if not spec.header_name:
            result.status = StepStatus.FAILED
            result.message = f"{side}_header: header_name is empty"
            return
        lower = {k.lower(): v for k, v in (headers or {}).items()}
        value = lower.get(spec.header_name.lower())
        passed = _compare(value, spec.expected, spec.op)
        result.status = StepStatus.PASSED if passed else StepStatus.FAILED
        result.message = (
            f"{side} header {spec.header_name} {spec.op.value} "
            f"{spec.expected!r}; got {value!r}"
        )

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    @staticmethod
    async def _check_schema(
        response: Response, spec: ApiAssertionSpec, result: AssertionResult
    ) -> None:
        from jsonschema import Draft7Validator

        if not spec.expected_schema:
            result.status = StepStatus.FAILED
            result.message = "response_schema: expected_schema is empty"
            return
        try:
            body = await response.json()
        except Exception as e:
            result.status = StepStatus.FAILED
            result.message = f"response_schema: response not JSON: {e}"
            return
        validator = Draft7Validator(spec.expected_schema)
        errors = sorted(validator.iter_errors(body), key=lambda e: list(e.path))
        if not errors:
            result.status = StepStatus.PASSED
            result.message = "response_schema: valid"
        else:
            msgs = [
                f"{'.'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
                for e in errors[:5]
            ]
            result.status = StepStatus.FAILED
            result.message = "response_schema: " + "; ".join(msgs)

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    @staticmethod
    def _check_time(
        response: Response, spec: ApiAssertionSpec, result: AssertionResult
    ) -> None:
        timing = getattr(response.request, "timing", None) or {}
        end = float(timing.get("responseEnd") or 0.0)
        start = float(timing.get("startTime") or 0.0)
        duration = max(0, int(end - start))
        passed = _compare(duration, spec.expected, spec.op)
        result.status = StepStatus.PASSED if passed else StepStatus.FAILED
        result.message = (
            f"response_time_ms {spec.op.value} {spec.expected}; got {duration}ms"
        )


# ---------------------------------------------------------------------
# Shared operator comparator
# ---------------------------------------------------------------------

def _compare(actual: Any, expected: str, op: ApiAssertionOp) -> bool:
    """Generic operator evaluation. Treats both sides as strings unless op is numeric."""
    a = "" if actual is None else str(actual)
    e = expected

    if op is ApiAssertionOp.EQUALS:
        return a == e
    if op is ApiAssertionOp.NOT_EQUALS:
        return a != e
    if op is ApiAssertionOp.CONTAINS:
        return e in a
    if op is ApiAssertionOp.NOT_CONTAINS:
        return e not in a
    if op is ApiAssertionOp.MATCHES_REGEX:
        return re.search(e, a) is not None
    if op is ApiAssertionOp.EXISTS:
        return actual is not None and a != ""
    if op is ApiAssertionOp.NOT_EXISTS:
        return actual is None or a == ""
    if op in (ApiAssertionOp.GT, ApiAssertionOp.GTE, ApiAssertionOp.LT, ApiAssertionOp.LTE):
        try:
            af = float(a)
            ef = float(e)
        except ValueError:
            return False
        if op is ApiAssertionOp.GT:
            return af > ef
        if op is ApiAssertionOp.GTE:
            return af >= ef
        if op is ApiAssertionOp.LT:
            return af < ef
        return af <= ef
    return False
