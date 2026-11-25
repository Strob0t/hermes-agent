"""Tests of the repeated-result guard metric (shared_metrics_harness._GUARD_DETECTORS).

The patch maps the guardrail codes ``repeated_result_halt`` and ``repeated_result_warning`` to the
existing loop-guard detector ``idempotent_no_progress`` (identical results without progress). No
new detector value is added, so the ingest contract stays the same. The tests show:
 - a ``halt`` verdict with ``repeated_result_halt`` gives 1 ``LOOP_GUARD_MARK`` with detector
   ``idempotent_no_progress`` and signal ``loop_detected``;
 - a ``warn`` verdict with ``repeated_result_warning`` gives signal ``repeated_tool_call``;
 - the same code 2 times in 1 turn gives only 1 mark.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.observability import shared_metrics_harness as harness


class RepeatedResultGuardMetricTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        for target, kwargs in (
            ("_active", {"return_value": True}),
            ("_record", {"side_effect": lambda attr, data: self.calls.append((attr, data))}),
        ):
            p = patch.object(harness, target, **kwargs)
            p.start()
            self.addCleanup(p.stop)
        self.agent = SimpleNamespace(provider="openrouter", model="anthropic/claude-sonnet")

    def _guard_marks(self):
        return [data for attr, data in self.calls if attr == "LOOP_GUARD_MARK"]

    def test_halt_code_gives_loop_detected_mark(self):
        harness.record_guardrail_decision(self.agent, "halt", "repeated_result_halt")
        marks = self._guard_marks()
        self.assertEqual(len(marks), 1)
        self.assertEqual(marks[0]["detector"], "idempotent_no_progress")
        self.assertEqual(marks[0]["signal"], "loop_detected")

    def test_warn_code_gives_repeated_tool_call_mark(self):
        harness.record_guardrail_decision(self.agent, "warn", "repeated_result_warning")
        marks = self._guard_marks()
        self.assertEqual(len(marks), 1)
        self.assertEqual(marks[0]["detector"], "idempotent_no_progress")
        self.assertEqual(marks[0]["signal"], "repeated_tool_call")

    def test_same_code_twice_in_one_turn_gives_one_mark(self):
        harness.record_guardrail_decision(self.agent, "halt", "repeated_result_halt")
        harness.record_guardrail_decision(self.agent, "halt", "repeated_result_halt")
        self.assertEqual(len(self._guard_marks()), 1)


if __name__ == "__main__":
    unittest.main()
