"""Tests of the local stop-hook patch (patch_stop_hook.py), run in the image build.

The patch adds a generic ``pre_stop`` plugin hook at the text-answer stop gate, with no file-edit
condition and its own bound ``agent.max_stop_continuations`` (default 20). The tests show:
 - with a registered ``pre_stop`` hook, ``apply_stop_gates`` continues the turn with no file edit;
 - with no hook (``has_hook`` False), the gate is a no-op and the turn finishes unchanged;
 - the bound ``max_stop_continuations`` reads the config and stops the loop;
 - ``get_pre_stop_continue_message`` returns the first non-empty continue/block message.
"""
import unittest
from unittest.mock import patch

import agent.turn_stop_gates as tsg
from agent.turn_stop_gates import StopGateVerdict, apply_stop_gates
from agent.verify_hooks import DEFAULT_MAX_STOP_CONTINUATIONS, max_stop_continuations
from hermes_cli.plugins import get_pre_stop_continue_message


class FakeAgent:
    """Minimal agent for the stop-gate loop: records the rows the gate flushes and appends."""

    def __init__(self, session_id="s1", platform="api_server"):
        self.session_id = session_id
        self.platform = platform
        self._session_messages = []
        self.interim = []
        self.flushed = 0

    def _emit_interim_assistant_message(self, msg):
        self.interim.append(msg)

    def _flush_messages_to_session_db(self, messages, history):
        self.flushed += 1

    def _interim_content_was_streamed(self, text):
        return False


def run_gates(agent, messages):
    """Run apply_stop_gates with the four earlier gates neutralized, so only pre_stop can fire."""
    final_msg = {"role": "assistant", "content": "done"}
    with patch.object(tsg, "_late_lsp_verdicts", return_value=None), \
         patch.object(tsg, "_verify_on_stop_nudge", return_value=None), \
         patch.object(tsg, "_pre_verify_nudge", return_value=None), \
         patch.object(tsg, "_kanban_stop_nudge", return_value=None):
        return apply_stop_gates(
            agent, final_msg, final_response="done", messages=messages,
            conversation_history=[], pending_verification_response="done",
            pending_verification_response_previewed=False,
        )


class BoundTest(unittest.TestCase):
    def test_default_bound_is_twenty(self):
        self.assertEqual(DEFAULT_MAX_STOP_CONTINUATIONS, 20)
        self.assertEqual(max_stop_continuations({}), 20)

    def test_config_key_overrides_the_bound(self):
        self.assertEqual(max_stop_continuations({"agent": {"max_stop_continuations": 3}}), 3)

    def test_bad_value_falls_back_to_default(self):
        self.assertEqual(max_stop_continuations({"agent": {"max_stop_continuations": "x"}}), 20)

    def test_negative_value_is_clamped_to_zero(self):
        self.assertEqual(max_stop_continuations({"agent": {"max_stop_continuations": -5}}), 0)


class DispatchTest(unittest.TestCase):
    def test_continue_message_wins(self):
        with patch("hermes_cli.plugins.invoke_hook",
                   return_value=[{"action": "continue", "message": " keep going "}]):
            self.assertEqual(get_pre_stop_continue_message(), "keep going")

    def test_claude_code_block_reason_wins(self):
        with patch("hermes_cli.plugins.invoke_hook",
                   return_value=[{"decision": "block", "reason": "not done"}]):
            self.assertEqual(get_pre_stop_continue_message(), "not done")

    def test_other_return_finishes(self):
        with patch("hermes_cli.plugins.invoke_hook", return_value=[{"action": "noop"}, "x", None]):
            self.assertIsNone(get_pre_stop_continue_message())

    def test_first_non_empty_message_wins(self):
        with patch("hermes_cli.plugins.invoke_hook",
                   return_value=[{"action": "continue", "message": "  "},
                                 {"action": "continue", "message": "second"}]):
            self.assertEqual(get_pre_stop_continue_message(), "second")


class GateContinueTest(unittest.TestCase):
    def test_pre_stop_continues_the_turn_without_a_file_edit(self):
        agent = FakeAgent()
        messages = [{"role": "user", "content": "do it"}]
        with patch("hermes_cli.lifecycle.has_hook", return_value=True), \
             patch("hermes_cli.plugins.get_pre_stop_continue_message", return_value="continue please"), \
             patch("agent.verify_hooks.max_stop_continuations", return_value=20):
            verdict = run_gates(agent, messages)
        self.assertTrue(verdict.continue_turn)
        self.assertIsNone(verdict.final_response)
        self.assertEqual(verdict.pending_verification_response, "done")
        self.assertEqual(agent._pre_stop_nudges, 1)
        # the answer row is flushed as interim, then a user-role nudge row is appended
        # (append_message may stamp extra fields such as timestamp; check the ones we set)
        nudge = messages[-1]
        self.assertEqual(nudge["role"], "user")
        self.assertEqual(nudge["content"], "continue please")
        self.assertIs(nudge["_pre_stop_synthetic"], True)
        self.assertEqual(agent.interim[-1]["content"], "done")

    def test_bound_stops_the_loop(self):
        agent = FakeAgent()
        agent._pre_stop_nudges = 2
        messages = [{"role": "user", "content": "do it"}]
        with patch("hermes_cli.lifecycle.has_hook", return_value=True), \
             patch("hermes_cli.plugins.get_pre_stop_continue_message", return_value="again"), \
             patch("agent.verify_hooks.max_stop_continuations", return_value=2):
            verdict = run_gates(agent, messages)
        # attempt (2) is not < bound (2): the gate does not fire
        self.assertFalse(verdict.continue_turn)
        self.assertEqual(agent._pre_stop_nudges, 2)


class GateNoOpTest(unittest.TestCase):
    def test_no_plugin_hook_means_no_behaviour_change(self):
        """With no pre_stop hook registered (has_hook False), the gate is a no-op: the turn finishes
        with the original final_response and nothing is appended or flushed."""
        agent = FakeAgent()
        messages = [{"role": "user", "content": "do it"}]
        with patch("hermes_cli.lifecycle.has_hook", return_value=False):
            verdict = run_gates(agent, messages)
        self.assertFalse(verdict.continue_turn)
        self.assertEqual(verdict.final_response, "done")
        self.assertEqual(len(messages), 1)
        self.assertEqual(agent.interim, [])
        self.assertEqual(agent.flushed, 0)
        self.assertFalse(hasattr(agent, "_pre_stop_nudges"))


if __name__ == "__main__":
    unittest.main()
