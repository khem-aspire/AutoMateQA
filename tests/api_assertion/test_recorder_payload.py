"""Recorder must translate an api_call payload into Assertion.api_spec."""

import unittest

from engine.models import Action, ActionType, AssertionType, ElementFingerprint, TestModel, TestStep
from engine.recorder import RecorderEngine


class TestRecorderApiPayload(unittest.TestCase):
    def setUp(self):
        self.model = TestModel()
        # Seed a step so handle_assertion attaches to "latest"
        self.model.steps.append(TestStep(
            step_id=1,
            action=Action(action_type=ActionType.CLICK),
            target=ElementFingerprint(),
        ))
        self.rec = RecorderEngine(self.model)
        self.rec._recording = True  # bypass start/stop for unit test

    def test_api_call_payload_creates_api_spec(self):
        payload = {
            "assertion_type": "api_call",
            "fingerprint": {},
            "api_spec": {
                "method": "POST",
                "path_template": "/api/users/{id}/balance",
                "query_keys_present": ["include"],
                "target": "response_jsonpath",
                "op": "equals",
                "expected": "100",
                "jsonpath": "$.balance",
                "header_name": "",
                "expected_schema": {},
            },
        }
        self.rec.handle_assertion(payload)
        step = self.model.steps[-1]
        self.assertEqual(len(step.assertions), 1)
        a = step.assertions[0]
        self.assertEqual(a.assertion_type, AssertionType.API_CALL)
        self.assertIsNotNone(a.api_spec)
        self.assertEqual(a.api_spec.path_template, "/api/users/{id}/balance")
        self.assertEqual(a.api_spec.jsonpath, "$.balance")
        self.assertEqual(a.api_spec.expected, "100")


if __name__ == "__main__":
    unittest.main()
