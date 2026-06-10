"""Tests for API-assertion model extensions."""

import json
import unittest

from engine.models import (
    ApiAssertionOp,
    ApiAssertionSpec,
    ApiAssertionTarget,
    Assertion,
    AssertionType,
    EngineConfig,
)


class TestApiAssertionModel(unittest.TestCase):
    def test_assertion_type_api_call_exists(self):
        self.assertEqual(AssertionType.API_CALL.value, "api_call")

    def test_api_spec_defaults(self):
        spec = ApiAssertionSpec()
        self.assertEqual(spec.method, "GET")
        self.assertEqual(spec.target, ApiAssertionTarget.STATUS)
        self.assertEqual(spec.op, ApiAssertionOp.EQUALS)
        self.assertEqual(spec.path_template, "")
        self.assertEqual(spec.query_keys_present, [])
        self.assertEqual(spec.jsonpath, "")
        self.assertEqual(spec.header_name, "")
        self.assertEqual(spec.expected_schema, {})

    def test_api_spec_roundtrip(self):
        spec = ApiAssertionSpec(
            method="POST",
            path_template="/api/users/{id}/balance",
            query_keys_present=["include"],
            target=ApiAssertionTarget.RESPONSE_JSONPATH,
            op=ApiAssertionOp.EQUALS,
            expected="100",
            jsonpath="$.balance",
        )
        data = json.loads(spec.model_dump_json())
        round_tripped = ApiAssertionSpec.model_validate(data)
        self.assertEqual(round_tripped, spec)

    def test_assertion_holds_api_spec(self):
        a = Assertion(
            assertion_type=AssertionType.API_CALL,
            api_spec=ApiAssertionSpec(
                method="GET",
                path_template="/api/x",
                target=ApiAssertionTarget.STATUS,
                op=ApiAssertionOp.EQUALS,
                expected="200",
            ),
        )
        data = json.loads(a.model_dump_json())
        round_tripped = Assertion.model_validate(data)
        self.assertEqual(round_tripped.assertion_type, AssertionType.API_CALL)
        self.assertIsNotNone(round_tripped.api_spec)
        self.assertEqual(round_tripped.api_spec.path_template, "/api/x")

    def test_assertion_api_spec_optional_default_none(self):
        a = Assertion()
        self.assertIsNone(a.api_spec)


class TestEngineConfigApiFields(unittest.TestCase):
    def test_defaults(self):
        cfg = EngineConfig()
        self.assertTrue(cfg.api_assertions_enabled)
        self.assertEqual(cfg.api_assertion_match_timeout_ms, 10_000)
        self.assertTrue(cfg.redact_in_ui)


if __name__ == "__main__":
    unittest.main()
