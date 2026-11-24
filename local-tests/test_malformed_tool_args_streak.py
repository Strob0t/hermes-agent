"""Tests of the malformed tool-argument streak port (fork fix/malformed-tool-args-streak).

``validate_tool_calls`` treated valid JSON that is not an object (a list, number or string) as
good arguments, so the executor refused each call and the model looped until the iteration
budget was gone. The patch routes non-object arguments through the bounded invalid-JSON retry,
counts the recovery rounds per turn in ``_malformed_args_recoveries``, and after 2 rounds ends
the turn with a typed, not-retryable ``malformed_tool_call`` partial exit. The counter is in
``_PER_TURN_RESET_STATE``. The tests show:
 - a list as tool arguments is malformed: no ``ok`` verdict, and the recovery error result
   says that the arguments must be a JSON object;
 - after the limit of consecutive malformed responses the turn ends with the
   ``malformed_tool_call`` message and ``failure_retryable`` False;
 - a new turn starts with the counter at 0, so one malformed response does not end it.
"""
import os
import tempfile
import unittest
from types import SimpleNamespace

from agent.turn_context import _reset_per_turn_agent_state
from agent.turn_failure_copy import site_copy
from agent.turn_tool_validation import validate_tool_calls


class FakeGuardrails:
    def reset_for_turn(self):
        pass


class FakeAgent:
    """Minimal agent for validate_tool_calls and the per-turn reset."""

    def __init__(self):
        self.valid_tool_names = {"read_file"}
        self.model = "fake-model"
        self.provider = "fake"
        self.log_prefix = ""
        self.tools = []
        self.api_mode = "anthropic_messages"
        self.max_iterations = 10
        self.run_budget_seconds = None
        self._compression_warning = None
        self._tool_guardrails = FakeGuardrails()
        self._memory_store = SimpleNamespace()
        self._invalid_tool_retries = 0
        self._invalid_json_retries = 0
        self.persisted = 0

    def _uniquify_tool_call_ids(self, tool_calls, taken=None):
        pass

    def _repair_tool_call(self, name):
        return None

    def _vprint(self, *a, **k):
        pass

    def _buffer_vprint(self, *a, **k):
        pass

    def _flush_status_buffer(self):
        pass

    def _cleanup_task_resources(self, task_id):
        pass

    def _persist_session(self, messages, history):
        self.persisted += 1

    def _build_assistant_message(self, msg, finish_reason):
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]}


_N = [0]


def _response(arguments):
    _N[0] += 1
    tc = SimpleNamespace(id=f"tc_{_N[0]}", call_id=None, type="function",
                         function=SimpleNamespace(name="read_file", arguments=arguments))
    return SimpleNamespace(tool_calls=[tc], provider_data=None)


def _validate(agent, messages, arguments):
    return validate_tool_calls(agent, _response(arguments), "tool_calls", messages=messages,
                               conversation_history=[], api_call_count=1, effective_task_id="t")


def _run_until_return(agent, messages, arguments, limit=50):
    for n in range(1, limit + 1):
        verdict = _validate(agent, messages, arguments)
        if verdict.action == "return":
            return n, verdict
        assert verdict.action == "continue", verdict.action
    return None, None


class _HomeCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = self._tmp.name

    def tearDown(self):
        if self._old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self._old_home
        self._tmp.cleanup()


class ListArgumentsAreMalformedTest(_HomeCase):
    def test_list_arguments_are_not_accepted(self):
        agent = FakeAgent()
        verdict = _validate(agent, [], ["a", "b"])
        self.assertNotEqual(verdict.action, "ok")

    def test_recovery_result_names_json_object(self):
        agent, messages = FakeAgent(), []
        for _ in range(3):
            self.assertEqual(_validate(agent, messages, '["a"]').action, "continue")
        tool_rows = [m for m in messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_rows), 1)
        self.assertIn("must be a JSON object", tool_rows[0]["content"])

    def test_object_arguments_still_pass(self):
        self.assertEqual(_validate(FakeAgent(), [], '{"path": "x"}').action, "ok")


class StreakLimitTest(_HomeCase):
    def test_consecutive_malformed_calls_end_the_turn(self):
        agent = FakeAgent()
        n, verdict = _run_until_return(agent, [], "[1, 2]")
        self.assertIsNotNone(n, "turn never ended on repeated non-object arguments")
        self.assertEqual(n, 9)
        self.assertEqual(verdict.result["failure_reason"], "malformed_tool_call")
        self.assertFalse(verdict.result["failure_retryable"])
        self.assertEqual(verdict.result["final_response"], site_copy("malformed_tool_call", model="fake-model"))
        self.assertFalse(verdict.result["completed"])


class NewTurnResetTest(_HomeCase):
    def test_new_turn_starts_with_counter_zero(self):
        agent = FakeAgent()
        for _ in range(8):
            self.assertEqual(_validate(agent, [], "[]").action, "continue")
        _reset_per_turn_agent_state(agent)
        self.assertEqual(agent._malformed_args_recoveries, 0)
        n, _ = _run_until_return(agent, [], "[]")
        self.assertEqual(n, 9)


if __name__ == "__main__":
    unittest.main()
