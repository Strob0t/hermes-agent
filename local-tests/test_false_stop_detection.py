"""Tests of the opt-in post-tool-round false-stop detection (port of fork finding #115).

After a tool round, a model sometimes stops (``finish_reason=stop``, no tool call) on text that
announces the next action ("Next I will run the tests:"), so the turn ends without the call. The
port adds ``agent/false_stop.py`` (detector) and one more continuation kind in
``agent/turn_final_response.py``: the reply stays as an interim row, a synthetic user nudge
follows and the loop continues. The budget is 2 nudges; ``agent/turn_tool_round.py`` resets it
after each tool round. Key ``agent.false_stop_detection`` (default ``off``). The tests show:
 - default config and explicit ``off``: the false stop ends the turn as before (no extra API call);
 - ``on``: 1 nudge and the loop continues; at most 2 nudges, then the turn ends on the reply;
 - ``on``: a final answer without intent markers is not nudged;
 - ``on``: no tool round in the turn -> no nudge;
 - ``on``: the budget resets after a new tool round (2 + 2 nudges in one turn);
 - ``auto``: off on a messaging surface, on for the CLI.

The provider client is a mock (no network); every test has its own temp HERMES_HOME.
"""
import os
import shutil
import tempfile
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_HOME = tempfile.mkdtemp(prefix="hermes-false-stop-")
os.environ["HERMES_HOME"] = _HOME

from run_agent import AIAgent  # noqa: E402

NUDGE_HEAD = "Your previous response ended as if you were about to call a tool"
FALSE_STOP = "Next I will run the tests:"
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


class FalseStopCase(unittest.TestCase):
    mode = None

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="hermes-false-stop-")
        env = patch.dict(os.environ, {"HERMES_HOME": self.home})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(shutil.rmtree, self.home, True)
        for var in ("HERMES_PLATFORM", "HERMES_SESSION_PLATFORM", "HERMES_SESSION_SOURCE"):
            os.environ.pop(var, None)
        if self.mode is not None:
            with open(os.path.join(self.home, "config.yaml"), "w") as fh:
                fh.write(f'agent:\n  false_stop_detection: "{self.mode}"\n')
        with patch("model_tools.get_tool_definitions", return_value=_tool_defs(TOOLS)), \
                patch("model_tools.check_toolset_requirements", return_value={}), \
                patch("agent.process_bootstrap.OpenAI"):
            agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                            max_iterations=20, quiet_mode=True, skip_context_files=True,
                            skip_memory=True, platform="cli")
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        self.agent = agent

    def run_turn(self, stages, user="fix the bug", history=None):
        create = self.agent.client.chat.completions.create
        create.reset_mock()
        create.side_effect = stages
        with patch.object(self.agent, "_persist_session"), \
                patch.object(self.agent, "_save_trajectory"), \
                patch.object(self.agent, "_cleanup_task_resources"), \
                patch("model_tools.handle_function_call", return_value="result: ok"):
            result = self.agent.run_conversation(user, conversation_history=history)
        return result, create.call_count

    def nudges(self, result):
        return [m for m in result["messages"]
                if m.get("role") == "user" and str(m.get("content", "")).startswith(NUDGE_HEAD)]


class DefaultConfig(FalseStopCase):
    def test_false_stop_after_tool_round_ends_turn(self):
        result, calls = self.run_turn([_tool_round(), _resp(FALSE_STOP), _resp("unused")])
        self.assertEqual(calls, 2)
        self.assertEqual(result["final_response"], FALSE_STOP)
        self.assertEqual(self.nudges(result), [])


class ExplicitOff(DefaultConfig):
    mode = "off"


class AutoOnMessagingSurface(DefaultConfig):
    mode = "auto"

    def setUp(self):
        super().setUp()
        env = patch.dict(os.environ, {"HERMES_PLATFORM": "telegram"})
        env.start()
        self.addCleanup(env.stop)


class Enabled(FalseStopCase):
    mode = "on"

    def test_false_stop_nudges_once_and_continues(self):
        result, calls = self.run_turn([_tool_round(), _resp(FALSE_STOP), _tool_round(),
                                       _resp("All 12 tests pass.")])
        self.assertEqual(calls, 4)
        self.assertEqual(result["final_response"], "All 12 tests pass.")
        self.assertEqual(len(self.nudges(result)), 1)

    def test_chinese_narration_nudges(self):
        result, calls = self.run_turn([_tool_round(), _resp("接下来看日志。"), _resp("日志没有错误。")])
        self.assertEqual(calls, 3)
        self.assertEqual(result["final_response"], "日志没有错误。")
        self.assertEqual(len(self.nudges(result)), 1)

    def test_at_most_two_nudges(self):
        result, calls = self.run_turn([_tool_round(), _resp(FALSE_STOP), _resp(FALSE_STOP),
                                       _resp(FALSE_STOP), _resp("unused")])
        self.assertEqual(calls, 4)
        self.assertEqual(result["final_response"], FALSE_STOP)
        self.assertEqual(len(self.nudges(result)), 2)

    def test_final_answer_without_intent_markers_not_nudged(self):
        for answer in ("The file has 3 lines.", "Result: the build is green.",
                       "Fixed: the bug is now gone, next release ships it."):
            with self.subTest(answer=answer):
                result, calls = self.run_turn([_tool_round(), _resp(answer), _resp("unused")])
                self.assertEqual(calls, 2)
                self.assertEqual(result["final_response"], answer)
                self.assertEqual(self.nudges(result), [])

    def test_no_tool_round_in_turn_not_nudged(self):
        history = [
            {"role": "user", "content": "q0"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_old", "type": "function",
                                                                "function": {"name": "web_search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_old", "content": "old result"},
            {"role": "assistant", "content": "a0"},
        ]
        result, calls = self.run_turn([_resp(FALSE_STOP), _resp("unused")], history=history)
        self.assertEqual(calls, 1)
        self.assertEqual(result["final_response"], FALSE_STOP)
        self.assertEqual(self.nudges(result), [])

    def test_budget_resets_after_new_tool_round(self):
        result, calls = self.run_turn([_tool_round(), _resp(FALSE_STOP), _resp(FALSE_STOP),
                                       _tool_round(), _resp(FALSE_STOP), _resp(FALSE_STOP),
                                       _resp("Done: tests pass.")])
        self.assertEqual(calls, 7)
        self.assertEqual(result["final_response"], "Done: tests pass.")
        self.assertEqual(len(self.nudges(result)), 4)


class AutoOnCli(FalseStopCase):
    mode = "auto"

    def test_false_stop_nudged_on_cli(self):
        result, calls = self.run_turn([_tool_round(), _resp(FALSE_STOP), _resp("Tests pass.")])
        self.assertEqual(calls, 3)
        self.assertEqual(result["final_response"], "Tests pass.")
        self.assertEqual(len(self.nudges(result)), 1)


if __name__ == "__main__":
    unittest.main()
