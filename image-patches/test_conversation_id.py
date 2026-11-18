"""Tests of the local conversation-id patch (patch_conversation_id.py), run in the image build.

The patch hands the compression-stable ``conversation_id`` to the ``pre_stop`` and
``transform_llm_output`` hooks, so a plugin can keep per-conversation state across a compaction that
rotates ``session_id``. The tests show:
 - ``get_pre_stop_continue_message`` forwards ``conversation_id`` to the hook;
 - the real ``_pre_stop_nudge`` gate passes the agent's conversation root (not its session id);
 - the real ``apply_llm_output_transform`` puts ``conversation_id`` in the hook payload;
 - a failed lookup gives an empty string and never raises into the turn.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import agent.turn_finalizer as tf
import agent.turn_stop_gates as tsg
from hermes_cli import plugins as plugins_facade


class RotatedAgent(SimpleNamespace):
    """An agent after a compaction: a new session_id, the same conversation root."""


def rotated_agent():
    return RotatedAgent(session_id="child-sess", model="m", platform="local",
                        _inherited_conversation_id="conv-root")


class PreStopDispatcherTest(unittest.TestCase):
    def test_conversation_id_reaches_the_hook(self):
        seen = {}

        def fake_invoke(name, **kw):
            seen.update(kw)
            return [{"action": "continue", "message": "go on"}]

        with patch.object(plugins_facade, "invoke_hook", fake_invoke):
            msg = plugins_facade.get_pre_stop_continue_message(
                session_id="child-sess", platform="local", attempt=0, final_response="x",
                conversation_id="conv-root")
        self.assertEqual(msg, "go on")
        self.assertEqual(seen["conversation_id"], "conv-root")
        self.assertEqual(seen["session_id"], "child-sess")  # the old fields are unchanged

    def test_conversation_id_is_optional(self):
        seen = {}

        def fake_invoke(name, **kw):
            seen.update(kw)
            return []

        with patch.object(plugins_facade, "invoke_hook", fake_invoke):
            self.assertIsNone(plugins_facade.get_pre_stop_continue_message(session_id="s"))
        self.assertEqual(seen["conversation_id"], "")


class GateTest(unittest.TestCase):
    def test_gate_passes_the_conversation_root_not_the_session_id(self):
        captured = {}

        def fake_msg(**kw):
            captured.update(kw)
            return "continue please"

        with patch("hermes_cli.lifecycle.has_hook", return_value=True), \
             patch.object(plugins_facade, "get_pre_stop_continue_message", fake_msg):
            out = tsg._pre_stop_nudge(rotated_agent(), "answer", 0)
        self.assertEqual(out, "continue please")
        self.assertEqual(captured["conversation_id"], "conv-root")
        self.assertEqual(captured["session_id"], "child-sess")

    def test_failed_lookup_gives_an_empty_string(self):
        with patch("agent.prompt_cache_scope.conversation_id", side_effect=RuntimeError("boom")):
            self.assertEqual(tsg._conversation_id_safe(rotated_agent()), "")


class TransformPayloadTest(unittest.TestCase):
    def test_payload_carries_conversation_id(self):
        seen = {}

        def fake_invoke(name, logger, **kw):
            seen.update(kw)
            return []

        agent = rotated_agent()
        with patch.object(tf, "_invoke_hook_safely", fake_invoke):
            tf.apply_llm_output_transform(agent, "MODEL REPORT", turn_id="t1", logger=SimpleNamespace())
        self.assertEqual(seen["conversation_id"], "conv-root")
        self.assertEqual(seen["session_id"], "child-sess")
        self.assertEqual(seen["response_text"], "MODEL REPORT")


if __name__ == "__main__":
    unittest.main()
