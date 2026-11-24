"""Tests of the opt-in post-tool placeholder nudge (port of fork finding #114).

Some models answer a tool result with only a progress marker ("working on it", "on it",
"正在处理") and stop. The port adds one more detector to ``agent/turn_final_response.py`` under
the shared continuation counter of the stall-intent / degenerate-final / ack guards (max 2 per
turn). Key ``agent.post_tool_placeholder_nudge`` (default ``false``). The tests show:
 - default config: a placeholder after a tool ends the turn as before (no extra API call);
 - enabled: 1 synthetic nudge row, the loop continues, at most 1 nudge per turn;
 - enabled: the shared counter limit is kept (2 stall nudges first -> no placeholder nudge);
 - enabled: no tool row in the last 6 messages -> no nudge;
 - enabled: a reply of only "ok" -> no nudge;
 - enabled: the flag resets at the next turn (``_PER_TURN_RESET_STATE``).

The provider client is a mock (no network); every test has its own temp HERMES_HOME.
"""
import os
import shutil
import tempfile
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_HOME = tempfile.mkdtemp(prefix="hermes-placeholder-")
os.environ["HERMES_HOME"] = _HOME

from run_agent import AIAgent  # noqa: E402

NUDGE_HEAD = "You returned a status message instead of the actual result."
TOOLS = ("web_search",)


def _tool_defs(names):
    return [{"type": "function", "function": {"name": n, "description": n,
                                              "parameters": {"type": "object", "properties": {}}}} for n in names]


def _resp(content, finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)],
                           model="test/model", usage=None)


def _tool_round():
    call = SimpleNamespace(id=f"call_{uuid.uuid4().hex[:8]}", type="function",
                           function=SimpleNamespace(name="web_search", arguments="{}"))
    return _resp("", "tool_calls", [call])


class PlaceholderCase(unittest.TestCase):
    enabled = None

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="hermes-placeholder-")
        env = patch.dict(os.environ, {"HERMES_HOME": self.home})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(shutil.rmtree, self.home, True)
        if self.enabled is not None:
            with open(os.path.join(self.home, "config.yaml"), "w") as fh:
                fh.write(f"agent:\n  post_tool_placeholder_nudge: {str(self.enabled).lower()}\n")
        with patch("model_tools.get_tool_definitions", return_value=_tool_defs(TOOLS)), \
                patch("model_tools.check_toolset_requirements", return_value={}), \
                patch("agent.process_bootstrap.OpenAI"):
            agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                            max_iterations=10, quiet_mode=True, skip_context_files=True,
                            skip_memory=True, platform="cli")
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        self.agent = agent

    def run_turn(self, stages, user="find the file", history=None):
        create = self.agent.client.chat.completions.create
        create.reset_mock()
        create.side_effect = stages
        with patch.object(self.agent, "_persist_session"), \
                patch.object(self.agent, "_save_trajectory"), \
                patch.object(self.agent, "_cleanup_task_resources"), \
                patch("model_tools.handle_function_call", return_value="result: found"):
            result = self.agent.run_conversation(user, conversation_history=history)
        return result, create.call_count

    def nudges(self, result):
        return [m for m in result["messages"]
                if m.get("role") == "user" and str(m.get("content", "")).startswith(NUDGE_HEAD)]


class DefaultConfig(PlaceholderCase):
    def test_placeholder_after_tool_ends_turn(self):
        result, calls = self.run_turn([_tool_round(), _resp("working on it"), _resp("unused")])
        self.assertEqual(calls, 2)
        self.assertEqual(result["final_response"], "working on it")
        self.assertEqual(self.nudges(result), [])


class ExplicitOff(DefaultConfig):
    enabled = False


class Enabled(PlaceholderCase):
    enabled = True

    def test_placeholder_after_tool_nudges_once_and_continues(self):
        result, calls = self.run_turn([_tool_round(), _resp("Working on it..."), _resp("Found it: a.txt")])
        self.assertEqual(calls, 3)
        self.assertEqual(result["final_response"], "Found it: a.txt")
        self.assertEqual(len(self.nudges(result)), 1)

    def test_chinese_placeholder_nudges(self):
        result, calls = self.run_turn([_tool_round(), _resp("好的，正在处理"), _resp("完成")])
        self.assertEqual(calls, 3)
        self.assertEqual(result["final_response"], "完成")

    def test_at_most_one_nudge_per_turn(self):
        result, calls = self.run_turn([_tool_round(), _resp("on it"), _resp("on it"), _resp("unused")])
        self.assertEqual(calls, 3)
        self.assertEqual(result["final_response"], "on it")
        self.assertEqual(len(self.nudges(result)), 1)

    def test_shared_counter_limit_is_kept(self):
        stall = "Let me now check the file."
        result, calls = self.run_turn([_tool_round(), _resp(stall), _resp(stall), _tool_round(),
                                       _resp("working on it"), _resp("unused")])
        self.assertEqual(calls, 5)
        self.assertEqual(result["final_response"], "working on it")
        self.assertEqual(self.nudges(result), [])

    def test_no_tool_row_in_last_six_no_nudge(self):
        history = [
            {"role": "user", "content": "q0"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_old", "type": "function",
                                                                "function": {"name": "web_search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_old", "content": "old result"},
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
        ]
        result, calls = self.run_turn([_resp("working on it"), _resp("unused")], history=history)
        self.assertEqual(calls, 1)
        self.assertEqual(result["final_response"], "working on it")
        self.assertEqual(self.nudges(result), [])

    def test_ok_reply_no_nudge(self):
        result, calls = self.run_turn([_tool_round(), _resp("ok"), _resp("unused")])
        self.assertEqual(calls, 2)
        self.assertEqual(result["final_response"], "ok")
        self.assertEqual(self.nudges(result), [])

    def test_flag_resets_next_turn(self):
        first, calls = self.run_turn([_tool_round(), _resp("on it"), _resp("Turn 1 done.")])
        self.assertEqual(calls, 3)
        second, calls = self.run_turn([_tool_round(), _resp("on it"), _resp("Turn 2 done.")],
                                      user="and the next file", history=first["messages"])
        self.assertEqual(calls, 3)
        self.assertEqual(second["final_response"], "Turn 2 done.")


if __name__ == "__main__":
    unittest.main()
