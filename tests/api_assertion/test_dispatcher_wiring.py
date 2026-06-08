"""AssertionEngine must dispatch API_CALL to ApiAssertionEvaluator."""

import asyncio
import unittest
from unittest.mock import MagicMock

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

        response = make_mock_response(status=200, url="https://x.com/api/x")

        page = MagicMock()

        def _on(event, cb):
            if event == "response":
                asyncio.get_event_loop().call_soon(cb, response)
        page.on = MagicMock(side_effect=_on)
        page.remove_listener = MagicMock()

        async def _never_idle(*_a, **_kw):
            await asyncio.Event().wait()
        page.wait_for_load_state = _never_idle

        assertion = Assertion(
            assertion_type=AssertionType.API_CALL,
            api_spec=ApiAssertionSpec(path_template="/api/x", expected="200"),
        )
        result = run(ae.evaluate(page, assertion))
        self.assertEqual(result.status, StepStatus.PASSED)


if __name__ == "__main__":
    unittest.main()
