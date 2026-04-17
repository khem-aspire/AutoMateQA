"""AssertionEngine must dispatch API_CALL to ApiAssertionEvaluator."""

import unittest
from unittest.mock import AsyncMock, MagicMock

from engine.assertions import AssertionEngine
from engine.models import (
    ApiAssertionSpec,
    Assertion,
    AssertionType,
    EngineConfig,
    StepStatus,
)
from tests.api_assertion.conftest import run, make_mock_response


class TestDispatcherWiring(unittest.TestCase):
    def test_api_call_is_dispatched(self):
        config = EngineConfig()
        selector_engine = MagicMock()
        ae = AssertionEngine(config, selector_engine, healing_engine=None)

        page = MagicMock()
        page.wait_for_response = AsyncMock(
            return_value=make_mock_response(status=200, url="https://x.com/api/x")
        )
        page.wait_for_load_state = AsyncMock()
        page.on = MagicMock()
        page.remove_listener = MagicMock()

        assertion = Assertion(
            assertion_type=AssertionType.API_CALL,
            api_spec=ApiAssertionSpec(path_template="/api/x", expected="200"),
        )
        result = run(ae.evaluate(page, assertion))
        self.assertEqual(result.status, StepStatus.PASSED)


if __name__ == "__main__":
    unittest.main()
