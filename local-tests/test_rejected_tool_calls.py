"""Tests of the port of finding #3 (rejected or never-run tool calls are marked as not run).

The port adds ``agent/tool_rejection.py`` and gives a tool call that never ran a structured status.
The tests show:
 - a command that matches an ``approvals.deny`` pattern (approvals.mode "off") gives a terminal
   result with status ``rejected``, and its compressor summary says ``not run, re-issue`` and
   ``forbidden`` and does not say "ran";
 - cancelled and timeout calls give their status and ``not run, re-issue if still needed``;
 - a normal successful terminal result keeps its summary;
 - the plain ``Error executing tool`` text prefix stays, so the ACP adapter detects the failure.
   The ``acp`` package is not in the test venv, so the test reads the ACP check from the source
   of ``acp_adapter/tools.py`` and does not import it.
"""
import ast
import json
import os
import tempfile
import unittest

_HOME = tempfile.mkdtemp(prefix="hermes-home-")
os.environ["HERMES_HOME"] = _HOME
with open(os.path.join(_HOME, "config.yaml"), "w", encoding="utf-8") as fh:
    fh.write("approvals:\n  mode: \"off\"\n  deny:\n    - \"rm -rf /srv/*\"\n")

from agent.context_compressor import _summarize_tool_result  # noqa: E402
from agent.replay_cleanup import _orphan_recovery, _INTERRUPTED_NOTICES  # noqa: E402
from agent.tool_rejection import (  # noqa: E402
    CANCELLED_STATUS,
    ERROR_EXECUTING_TOOL_PREFIX,
    REJECTED_STATUS,
    TIMEOUT_STATUS,
    outcome_result,
)
from agent.tool_executor import _ToolCallRef, _unfinished_tool_result  # noqa: E402
from tools import terminal_tool  # noqa: E402

DENIED = "rm -rf /srv/data"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeAgent:
    """Only the attributes ``_unfinished_tool_result`` reads."""

    def __init__(self, interrupted):
        self._interrupt_requested = interrupted
        self._tool_interrupt_reason = None


def unfinished(interrupted, timed_out):
    ref = _ToolCallRef("terminal", {"command": "make build"}, "task-1", "call-1", [])
    return _unfinished_tool_result(FakeAgent(interrupted), ref, timed_out=timed_out, timeout_s=30.0)[0]


def terminal_args(command):
    return json.dumps({"command": command})


class DenyPathTest(unittest.TestCase):
    def denied_result(self):
        with self.assertRaises(terminal_tool._Rejected) as ctx:
            terminal_tool._run_approval_guards(DENIED, "local", {}, force=False)
        return ctx.exception.result_json

    def test_deny_rule_gives_rejected_status(self):
        payload = json.loads(self.denied_result())
        self.assertEqual(payload["status"], REJECTED_STATUS)
        self.assertIn("approvals.deny", payload["error"])

    def test_deny_summary_says_not_run_and_forbidden(self):
        summary = _summarize_tool_result("terminal", terminal_args(DENIED), self.denied_result())
        self.assertIn("not run, re-issue", summary)
        self.assertIn("forbidden", summary)
        self.assertNotIn("ran `", summary)
        self.assertNotIn("if still needed", summary)

    def test_other_command_is_not_rejected(self):
        verdict = terminal_tool._run_approval_guards("echo hi", "local", {}, force=False)
        self.assertIsInstance(verdict, terminal_tool._ApprovalVerdict)


class NeverRunTest(unittest.TestCase):
    def test_interrupted_call_gets_cancelled_status(self):
        content = unfinished(interrupted=True, timed_out=False)
        self.assertEqual(json.loads(content)["status"], CANCELLED_STATUS)
        summary = _summarize_tool_result("terminal", terminal_args("make build"), content)
        self.assertIn("cancelled: not run, re-issue if still needed", summary)

    def test_timed_out_call_gets_timeout_summary(self):
        content = unfinished(interrupted=False, timed_out=True)
        summary = _summarize_tool_result("terminal", terminal_args("make build"), content)
        self.assertIn("timeout: not run, re-issue if still needed", summary)
        self.assertNotIn("ran `", summary)

    def test_cancelled_status_summary(self):
        content = outcome_result(CANCELLED_STATUS, "[Tool execution cancelled — terminal was skipped. stop]")
        self.assertEqual(json.loads(content)["status"], CANCELLED_STATUS)
        summary = _summarize_tool_result("terminal", terminal_args("make build"), content)
        self.assertIn("cancelled: not run, re-issue if still needed", summary)
        self.assertNotIn("ran `", summary)

    def test_timeout_summary(self):
        content = f"{ERROR_EXECUTING_TOOL_PREFIX} 'terminal': timed out after 30.0s"
        summary = _summarize_tool_result("terminal", terminal_args("make build"), content)
        self.assertIn(f"{TIMEOUT_STATUS}: not run, re-issue if still needed", summary)

    def test_timeout_status_payload_summary(self):
        content = outcome_result(TIMEOUT_STATUS, "worker gave up")
        summary = _summarize_tool_result("execute_code", json.dumps({"code": "x"}), content)
        self.assertIn("timeout: not run, re-issue if still needed", summary)

    def test_orphan_recovery_carries_status(self):
        disposition, content = _orphan_recovery("terminal", _INTERRUPTED_NOTICES)
        self.assertEqual(disposition, "unknown")
        self.assertEqual(json.loads(content)["status"], "unknown")
        summary = _summarize_tool_result("terminal", terminal_args("make build"), content)
        self.assertIn("unknown: not run, re-issue if still needed", summary)


class UnchangedTest(unittest.TestCase):
    def test_successful_result_summary_unchanged(self):
        content = json.dumps({"output": "a\nb", "exit_code": 0, "error": None})
        summary = _summarize_tool_result("terminal", terminal_args("ls"), content)
        self.assertEqual(summary, "[terminal] ran `ls` -> exit 0, 2 lines output")

    def test_error_executing_tool_prefix_unchanged(self):
        self.assertEqual(ERROR_EXECUTING_TOOL_PREFIX, "Error executing tool")
        content = unfinished(interrupted=False, timed_out=True)
        self.assertTrue(content.startswith("Error executing tool 'terminal': timed out after 30.0s"))
        with open(os.path.join(REPO, "acp_adapter", "tools.py"), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        failed = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_tool_result_failed")
        prefixes = [
            c.args[0].value for c in ast.walk(failed)
            if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "startswith"
            and c.args and isinstance(c.args[0], ast.Constant)
        ]
        self.assertIn("Error executing tool '", prefixes)
        self.assertTrue(any(content.startswith(p) for p in prefixes))


if __name__ == "__main__":
    unittest.main()
