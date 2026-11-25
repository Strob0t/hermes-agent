"""Tests of the repeated-result guard in agent/tool_guardrails.py (port of the source fork's
``_track_result_repetition``), run in the image build.

The patch counts consecutive identical tool results, whatever the tool name, arguments or failure
status. It warns at ``warn_after.repeated_result`` (default 3) and halts at
``hard_stop_after.repeated_result`` (default 5) when hard stops are active. Pollers are exempt:
``process_manage`` and terminal commands that match ``repeated_result_exempt_command_patterns``
(default: ``oh-task.py status|wait|list``). The tests show:
 - identical results for calls with different args warn and halt at the default thresholds;
 - changed thresholds in the config change the trigger point;
 - identical ``process_manage`` poll results and ``oh-task.py status`` terminal results never trigger;
 - different results never trigger.
"""
import os
import tempfile
import unittest

from unittest.mock import patch

_HOME = tempfile.mkdtemp(prefix="hermes-home-")
_ENV = patch.dict(os.environ, {"HERMES_HOME": _HOME})


def setUpModule():
    global ToolCallGuardrailConfig, ToolCallGuardrailController, DEFAULT_CONFIG
    _ENV.start()
    from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController
    from hermes_cli.config_defaults import DEFAULT_CONFIG


def tearDownModule():
    _ENV.stop()


BLOCKED_BODY = "Access denied: the remote site blocked this request. " * 8


def controller(**overrides):
    cfg = dict(DEFAULT_CONFIG["tool_loop_guardrails"])
    cfg["hard_stop_enabled"] = True
    cfg.update(overrides)
    return ToolCallGuardrailController(ToolCallGuardrailConfig.from_mapping(cfg, platform="cli"))


def run_calls(ctrl, calls):
    """Feed (tool, args, result) calls; return the action/code of each after_call decision."""
    return [(d.action, d.code) for d in (ctrl.after_call(t, a, r) for t, a, r in calls)]


def varied_fetches(n, result=BLOCKED_BODY):
    return [("execute_code", {"code": f"fetch('https://example.org/page/{i}')"}, result) for i in range(n)]


class RepeatedResultTriggerTest(unittest.TestCase):
    def test_default_thresholds_warn_at_3_and_halt_at_5_with_varied_args(self):
        ctrl = controller()
        actions = run_calls(ctrl, varied_fetches(5))
        self.assertEqual([a for a, _ in actions[:2]], ["allow", "allow"])
        self.assertEqual(actions[2], ("warn", "repeated_result_warning"))
        self.assertEqual(actions[3], ("warn", "repeated_result_warning"))
        self.assertEqual(actions[4], ("halt", "repeated_result_halt"))
        self.assertIsNotNone(ctrl.halt_decision)
        self.assertEqual(ctrl.halt_decision.code, "repeated_result_halt")

    def test_warn_only_session_never_halts(self):
        ctrl = controller(hard_stop_enabled=False)
        actions = run_calls(ctrl, varied_fetches(8))
        self.assertNotIn("halt", [a for a, _ in actions])
        self.assertEqual(actions[-1], ("warn", "repeated_result_warning"))


class RepeatedResultThresholdConfigTest(unittest.TestCase):
    def test_changed_thresholds_move_the_trigger_point(self):
        ctrl = controller(warn_after={"repeated_result": 5}, hard_stop_after={"repeated_result": 7})
        actions = run_calls(ctrl, varied_fetches(7))
        self.assertEqual([a for a, _ in actions[:4]], ["allow"] * 4)
        self.assertEqual(actions[4], ("warn", "repeated_result_warning"))
        self.assertEqual(actions[5], ("warn", "repeated_result_warning"))
        self.assertEqual(actions[6], ("halt", "repeated_result_halt"))


class RepeatedResultPollerExemptTest(unittest.TestCase):
    def test_identical_process_manage_polls_do_not_trigger(self):
        ctrl = controller()
        poll = '{"status": "running", "output": "' + "still building ... " * 20 + '"}'
        calls = [("process_manage", {"action": "poll", "session_id": "p1"}, poll)] * 10
        codes = [c for _, c in run_calls(ctrl, calls)]
        self.assertNotIn("repeated_result_warning", codes)
        self.assertNotIn("repeated_result_halt", codes)

    def test_identical_oh_task_status_results_do_not_trigger(self):
        ctrl = controller()
        status = "task x: state=running step=3/9 " * 12
        calls = [("terminal", {"command": "python3 -I /opt/data/scripts/oh-task.py status x"}, status)] * 10
        codes = [c for _, c in run_calls(ctrl, calls)]
        self.assertNotIn("repeated_result_warning", codes)
        self.assertNotIn("repeated_result_halt", codes)


class RepeatedResultDifferentResultsTest(unittest.TestCase):
    def test_different_results_do_not_trigger(self):
        ctrl = controller()
        calls = [
            ("execute_code", {"code": f"fetch({i})"}, f"page {i}: " + "content " * 40) for i in range(10)
        ]
        self.assertEqual([a for a, _ in run_calls(ctrl, calls)], ["allow"] * 10)
        self.assertIsNone(ctrl.halt_decision)


if __name__ == "__main__":
    unittest.main()
