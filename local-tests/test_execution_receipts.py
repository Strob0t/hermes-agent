"""Tests of the opt-in per-turn execution receipts (port of fork finding #101).

The port adds ``agent/execution_receipt.py``. With ``agent.execution_receipts.enabled`` (default
``false``) ``run_conversation`` in ``agent/conversation_loop.py`` runs each turn under one receipt and
finalizes it on every exit path. ``hermes_cli/middleware.py`` records each provider request and tool
dispatch that reaches the terminal callback of the execution middleware chain, with each frame's
decision. ``gateway/status.py`` gives the process identity in the receipt. The receipt is one owner-only
(0o600) JSON file below ``<HERMES_HOME>/runtime/turn-execution-receipts/``, written atomically, and an
existing receipt is never replaced. The second opt-in key, ``agent.oneshot_mcp_wait_seconds`` (default
``0``), lets a one-shot session wait for MCP discovery in flight before its first turn
(``hermes_cli/oneshot.py``). The tests show:
 - default config: a turn with a tool call writes no receipt file;
 - enabled: a turn with 1 tool call writes 1 receipt with both provider calls and the tool call, mode 0o600;
 - enabled: a turn that raises or is interrupted still finalizes its receipt (``failed`` / ``interrupted``);
 - enabled: an existing receipt file for the turn is not replaced, and no temp file stays;
 - ``oneshot_mcp_wait_seconds`` 0: no wait (also not for a one-shot agent); a positive value waits.

The provider client is a mock (no network); every test has its own temp HERMES_HOME.
"""
import json
import os
import shutil
import stat
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_HOME = tempfile.mkdtemp(prefix="hermes-receipts-")
os.environ["HERMES_HOME"] = _HOME

from run_agent import AIAgent  # noqa: E402

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


class ReceiptCase(unittest.TestCase):
    config = None

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="hermes-receipts-")
        env = patch.dict(os.environ, {"HERMES_HOME": self.home})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(shutil.rmtree, self.home, True)
        if self.config is not None:
            with open(os.path.join(self.home, "config.yaml"), "w") as fh:
                fh.write(self.config)
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
        self.tool_calls = 0

    def _tool(self, *args, **kwargs):
        self.tool_calls += 1
        return "result: ok"

    def run_turn(self, stages, tool=None, task_id=None):
        create = self.agent.client.chat.completions.create
        create.reset_mock()
        create.side_effect = stages
        with patch.object(self.agent, "_persist_session"), \
                patch.object(self.agent, "_save_trajectory"), \
                patch.object(self.agent, "_cleanup_task_resources"), \
                patch("model_tools.handle_function_call", side_effect=tool or self._tool):
            return self.agent.run_conversation("look it up", task_id=task_id)

    def receipt_root(self):
        return Path(self.home) / "runtime" / "turn-execution-receipts"

    def receipt_files(self):
        root = self.receipt_root()
        return sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []


class DefaultConfig(ReceiptCase):
    def test_turn_writes_no_receipt(self):
        result = self.run_turn([_tool_round(), _resp("done")])
        self.assertEqual(result["final_response"], "done")
        self.assertEqual(self.tool_calls, 1)
        self.assertFalse(self.receipt_root().exists())


class ExplicitOff(DefaultConfig):
    config = "agent:\n  execution_receipts:\n    enabled: false\n"


class Enabled(ReceiptCase):
    config = "agent:\n  execution_receipts:\n    enabled: true\n"

    def load_single_receipt(self):
        files = self.receipt_files()
        self.assertEqual(len(files), 1, files)
        return files[0], json.loads(files[0].read_text())

    def test_turn_with_tool_call_writes_one_owner_only_receipt(self):
        result = self.run_turn([_tool_round(), _resp("done")])
        self.assertEqual(result["final_response"], "done")
        path, receipt = self.load_single_receipt()
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(receipt["session_id"], self.agent.session_id)
        self.assertEqual(receipt["turn_id"], result["turn_id"])
        self.assertEqual(receipt["final_status"], "completed")
        self.assertEqual(receipt["provider_execution_count"], 2)
        self.assertEqual([e["status"] for e in receipt["provider_executions"]], ["returned", "returned"])
        self.assertEqual(receipt["tool_execution_count"], 1)
        tool = receipt["tool_executions"][0]
        self.assertEqual((tool["tool_name"], tool["status"]), ("web_search", "returned"))
        self.assertTrue(tool["tool_call_id"].startswith("tc-sha256:"))
        self.assertEqual([d["decision"] for d in receipt["request_decisions"]], ["executed", "executed"])
        self.assertEqual([d["decision"] for d in receipt["tool_decisions"]], ["executed"])
        self.assertEqual(receipt["execution_accounting"]["status"], "exact")

    def test_each_turn_writes_its_own_receipt(self):
        self.run_turn([_resp("one")])
        self.run_turn([_resp("two")])
        self.assertEqual(len(self.receipt_files()), 2)

    def test_turn_that_raises_still_finalizes_receipt(self):
        for error, status in ((RuntimeError("boom"), "failed"), (KeyboardInterrupt(), "interrupted")):
            with self.subTest(error=type(error).__name__):
                shutil.rmtree(self.receipt_root(), ignore_errors=True)
                with patch("agent.turn_finalizer.settle_held_final", side_effect=error):
                    with self.assertRaises(type(error)):
                        self.run_turn([_tool_round(), _resp("done")])
                _, receipt = self.load_single_receipt()
                self.assertEqual(receipt["final_status"], status)
                self.assertEqual(receipt["provider_execution_count"], 2)
                self.assertEqual(receipt["tool_execution_count"], 1)

    def test_existing_receipt_is_not_replaced(self):
        from agent.execution_receipt import get_turn_execution_receipt_path

        fixed = uuid.UUID("0123456789abcdef0123456789abcdef")
        turn_id = f"{self.agent.session_id}:fixed-task:{fixed.hex[:8]}"
        path = get_turn_execution_receipt_path(self.agent.session_id, turn_id)
        path.parent.mkdir(parents=True, mode=0o700)
        path.write_text("prior receipt\n")
        # uuid4 is fixed for this turn: the facade builds the turn id from task_id and uuid4().
        with patch("agent.turn_facade.uuid.uuid4", return_value=fixed):
            result = self.run_turn([_resp("done")], task_id="fixed-task")
        self.assertEqual(result["final_response"], "done")
        self.assertEqual(result["turn_id"], turn_id)
        self.assertEqual(path.read_text(), "prior receipt\n")
        self.assertEqual(self.receipt_files(), [path])


class OneshotMcpWait(ReceiptCase):
    def test_zero_does_not_wait(self):
        from hermes_cli.oneshot import wait_for_oneshot_mcp_discovery

        with patch("hermes_cli.mcp_startup.wait_for_mcp_discovery") as wait:
            self.assertFalse(wait_for_oneshot_mcp_discovery(0))
        wait.assert_not_called()

    def test_default_key_does_not_wait_on_oneshot_turn(self):
        self.agent._oneshot_session = True
        with patch("hermes_cli.mcp_startup.wait_for_mcp_discovery") as wait:
            result = self.run_turn([_resp("done")])
        self.assertEqual(result["final_response"], "done")
        wait.assert_not_called()


class OneshotMcpWaitOn(ReceiptCase):
    config = "agent:\n  oneshot_mcp_wait_seconds: 5\n"

    def test_positive_key_waits_on_oneshot_turn(self):
        self.agent._oneshot_session = True
        with patch("hermes_cli.mcp_startup.wait_for_mcp_discovery") as wait:
            self.run_turn([_resp("done")])
        wait.assert_called_once_with(timeout=5.0)

    def test_positive_key_does_not_wait_on_interactive_turn(self):
        self.agent._oneshot_session = False
        with patch("hermes_cli.mcp_startup.wait_for_mcp_discovery") as wait:
            self.run_turn([_resp("done")])
        wait.assert_not_called()


if __name__ == "__main__":
    unittest.main()
