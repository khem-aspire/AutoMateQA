"""Unit tests for ApiAssertionEvaluator."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from engine.api_assertion import ApiAssertionEvaluator
from engine.models import (
    ApiAssertionOp,
    ApiAssertionSpec,
    ApiAssertionTarget,
    Assertion,
    AssertionType,
    EngineConfig,
    StepStatus,
)

from tests.api_assertion.conftest import run, make_mock_response


def _spec(**overrides):
    base = dict(
        method="GET",
        path_template="/api/x",
        target=ApiAssertionTarget.STATUS,
        op=ApiAssertionOp.EQUALS,
        expected="200",
    )
    base.update(overrides)
    return ApiAssertionSpec(**base)


def _assertion(**overrides):
    return Assertion(
        assertion_type=AssertionType.API_CALL,
        api_spec=_spec(**overrides),
    )


def _page_with_response(response):
    page = MagicMock()
    page.wait_for_response = AsyncMock(return_value=response)
    page.wait_for_load_state = AsyncMock()
    page.on = MagicMock()
    page.remove_listener = MagicMock()
    return page


class TestStatusTarget(unittest.TestCase):
    def test_status_equals_pass(self):
        page = _page_with_response(make_mock_response(status=200, url="https://x.com/api/x"))
        ev = ApiAssertionEvaluator(EngineConfig())
        result = run(ev.evaluate(page, _assertion()))
        self.assertEqual(result.status, StepStatus.PASSED)

    def test_status_equals_fail(self):
        page = _page_with_response(make_mock_response(status=404, url="https://x.com/api/x"))
        ev = ApiAssertionEvaluator(EngineConfig())
        result = run(ev.evaluate(page, _assertion(expected="200")))
        self.assertEqual(result.status, StepStatus.FAILED)
        self.assertIn("404", result.message)

    def test_status_wildcard_2xx(self):
        page = _page_with_response(make_mock_response(status=201, url="https://x.com/api/x"))
        ev = ApiAssertionEvaluator(EngineConfig())
        result = run(ev.evaluate(page, _assertion(expected="2xx")))
        self.assertEqual(result.status, StepStatus.PASSED)

    def test_status_not_equals(self):
        page = _page_with_response(make_mock_response(status=500, url="https://x.com/api/x"))
        ev = ApiAssertionEvaluator(EngineConfig())
        result = run(ev.evaluate(page, _assertion(op=ApiAssertionOp.NOT_EQUALS, expected="200")))
        self.assertEqual(result.status, StepStatus.PASSED)


class TestResponseJsonpath(unittest.TestCase):
    def _eval(self, body, spec_kwargs):
        page = _page_with_response(
            make_mock_response(status=200, url="https://x.com/api/x", response_json=body)
        )
        ev = ApiAssertionEvaluator(EngineConfig())
        return run(ev.evaluate(page, _assertion(
            target=ApiAssertionTarget.RESPONSE_JSONPATH,
            **spec_kwargs,
        )))

    def test_equals_leaf(self):
        r = self._eval({"user": {"balance": 100}},
                       dict(jsonpath="$.user.balance", op=ApiAssertionOp.EQUALS, expected="100"))
        self.assertEqual(r.status, StepStatus.PASSED)

    def test_contains_string(self):
        r = self._eval({"msg": "hello world"},
                       dict(jsonpath="$.msg", op=ApiAssertionOp.CONTAINS, expected="hello"))
        self.assertEqual(r.status, StepStatus.PASSED)

    def test_regex_match(self):
        r = self._eval({"code": "ABC-123"},
                       dict(jsonpath="$.code", op=ApiAssertionOp.MATCHES_REGEX, expected=r"^[A-Z]{3}-\d+$"))
        self.assertEqual(r.status, StepStatus.PASSED)

    def test_gt_numeric(self):
        r = self._eval({"n": 10},
                       dict(jsonpath="$.n", op=ApiAssertionOp.GT, expected="5"))
        self.assertEqual(r.status, StepStatus.PASSED)

    def test_path_not_found(self):
        r = self._eval({"a": 1},
                       dict(jsonpath="$.missing", op=ApiAssertionOp.EQUALS, expected="x"))
        self.assertEqual(r.status, StepStatus.FAILED)
        self.assertIn("path", r.message.lower())

    def test_exists(self):
        r = self._eval({"a": 1},
                       dict(jsonpath="$.a", op=ApiAssertionOp.EXISTS, expected=""))
        self.assertEqual(r.status, StepStatus.PASSED)


class TestRequestJsonpath(unittest.TestCase):
    def test_request_body_present(self):
        page = _page_with_response(make_mock_response(
            method="POST", url="https://x.com/api/x",
            request_post_data_json={"user": {"email": "a@b.com"}},
        ))
        ev = ApiAssertionEvaluator(EngineConfig())
        result = run(ev.evaluate(page, _assertion(
            method="POST",
            target=ApiAssertionTarget.REQUEST_JSONPATH,
            jsonpath="$.user.email",
            op=ApiAssertionOp.EQUALS,
            expected="a@b.com",
        )))
        self.assertEqual(result.status, StepStatus.PASSED)

    def test_no_request_body(self):
        page = _page_with_response(make_mock_response(
            method="GET", url="https://x.com/api/x", request_post_data_json=None,
        ))
        ev = ApiAssertionEvaluator(EngineConfig())
        result = run(ev.evaluate(page, _assertion(
            target=ApiAssertionTarget.REQUEST_JSONPATH,
            jsonpath="$.anything",
        )))
        self.assertEqual(result.status, StepStatus.FAILED)
        self.assertIn("no request body", result.message)


class TestHeaders(unittest.TestCase):
    def test_response_header_case_insensitive(self):
        page = _page_with_response(make_mock_response(
            url="https://x.com/api/x",
            response_headers={"Content-Type": "application/json"},
        ))
        ev = ApiAssertionEvaluator(EngineConfig())
        result = run(ev.evaluate(page, _assertion(
            target=ApiAssertionTarget.RESPONSE_HEADER,
            header_name="content-type",
            op=ApiAssertionOp.CONTAINS,
            expected="json",
        )))
        self.assertEqual(result.status, StepStatus.PASSED)

    def test_request_header_missing(self):
        page = _page_with_response(make_mock_response(
            url="https://x.com/api/x", request_headers={"X-Other": "y"},
        ))
        ev = ApiAssertionEvaluator(EngineConfig())
        result = run(ev.evaluate(page, _assertion(
            target=ApiAssertionTarget.REQUEST_HEADER,
            header_name="authorization",
            op=ApiAssertionOp.EXISTS,
        )))
        self.assertEqual(result.status, StepStatus.FAILED)


class TestResponseSchema(unittest.TestCase):
    def _schema(self):
        return {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"},
            },
            "required": ["id", "name"],
        }

    def test_schema_valid(self):
        page = _page_with_response(make_mock_response(
            url="https://x.com/api/x", response_json={"id": 1, "name": "a"},
        ))
        ev = ApiAssertionEvaluator(EngineConfig())
        result = run(ev.evaluate(page, _assertion(
            target=ApiAssertionTarget.RESPONSE_SCHEMA,
            expected_schema=self._schema(),
        )))
        self.assertEqual(result.status, StepStatus.PASSED)

    def test_schema_invalid(self):
        page = _page_with_response(make_mock_response(
            url="https://x.com/api/x", response_json={"id": "wrong"},
        ))
        ev = ApiAssertionEvaluator(EngineConfig())
        result = run(ev.evaluate(page, _assertion(
            target=ApiAssertionTarget.RESPONSE_SCHEMA,
            expected_schema=self._schema(),
        )))
        self.assertEqual(result.status, StepStatus.FAILED)
        self.assertTrue(result.message)


class TestResponseTime(unittest.TestCase):
    def test_lt_pass(self):
        page = _page_with_response(make_mock_response(
            url="https://x.com/api/x", timing_total_ms=50.0,
        ))
        ev = ApiAssertionEvaluator(EngineConfig())
        result = run(ev.evaluate(page, _assertion(
            target=ApiAssertionTarget.RESPONSE_TIME_MS,
            op=ApiAssertionOp.LT, expected="200",
        )))
        self.assertEqual(result.status, StepStatus.PASSED)

    def test_gt_fail(self):
        page = _page_with_response(make_mock_response(
            url="https://x.com/api/x", timing_total_ms=30.0,
        ))
        ev = ApiAssertionEvaluator(EngineConfig())
        result = run(ev.evaluate(page, _assertion(
            target=ApiAssertionTarget.RESPONSE_TIME_MS,
            op=ApiAssertionOp.GT, expected="200",
        )))
        self.assertEqual(result.status, StepStatus.FAILED)


class TestMatchFailure(unittest.TestCase):
    def test_networkidle_without_match(self):
        page = MagicMock()
        never = asyncio.Event()

        async def _wfr(predicate):
            await never.wait()
        page.wait_for_response = _wfr
        page.wait_for_load_state = AsyncMock()
        page.on = MagicMock()
        page.remove_listener = MagicMock()
        ev = ApiAssertionEvaluator(EngineConfig(api_assertion_match_timeout_ms=500))
        ev._step_window = [
            {"method": "POST", "url": "https://x.com/api/profile", "status": 200, "duration_ms": 120},
        ]
        result = run(ev.evaluate(page, _assertion(
            method="POST", path_template="/api/users/{id}/balance",
        )))
        self.assertEqual(result.status, StepStatus.FAILED)
        self.assertIn("not seen before networkidle", result.message)
        self.assertIn("/api/profile", result.message)
        self.assertEqual(result.diagnostic["expected"]["path_template"], "/api/users/{id}/balance")


if __name__ == "__main__":
    unittest.main()
