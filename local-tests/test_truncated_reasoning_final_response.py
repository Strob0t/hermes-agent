"""End-to-end tests of the truncated-reasoning promotion fix through ``finish_text_response``.

The fix keeps a cut-off reasoning field from being delivered as the final answer of a
reasoning-only response. These tests call only ``agent.turn_final_response.finish_text_response``,
which exists before and after the fix, so the first test fails by assertion on a tree without it.
The agent is on the trusted local Nemotron parser route, where reasoning promotion is permitted.
The empty-response recovery, the stop gates and the output transform are patched, because they
do I/O outside this decision. The tests show:
 - finish_reason ``stop`` with reasoning that is cut off is not delivered as the final answer;
 - finish_reason ``stop`` with complete reasoning is delivered as the final answer;
 - finish_reason ``length`` with complete reasoning is not delivered as the final answer.
"""
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.agent_runtime_helpers import strip_think_blocks
from agent.turn_final_response import finish_text_response
from agent.turn_stop_gates import StopGateVerdict

ANSWER = "The migration is complete. All 12 tables were copied and the checksums match."
CUT = "The fix was already applied on 2026-"
EMPTY_PATH = "<empty-response path>"


class FakeAgent:
    """Minimal agent for the final-text branch on the trusted local Nemotron parser route."""

    model = "nemotron-3.5-lightning"
    provider = "custom"
    requested_provider = ""
    base_url = "http://127.0.0.1:8000/v1"
    api_mode = "chat_completions"
    _custom_providers = []
    _stall_guards = False
    valid_tool_names = []
    quiet_mode = True
    session_id = "s1"

    def __init__(self):
        self.flushed = []

    def _extract_reasoning(self, message):
        return message.reasoning

    def _strip_think_blocks(self, content):
        return strip_think_blocks(self, content)

    def _has_content_after_think_block(self, content):
        return bool(content) and bool(self._strip_think_blocks(content).strip())

    def _clear_status_buffer(self):
        pass

    def _build_assistant_message(self, assistant_message, finish_reason):
        return {"role": "assistant", "content": assistant_message.content or "",
                "reasoning": assistant_message.reasoning, "finish_reason": finish_reason}

    def _flush_messages_to_session_db(self, messages, conversation_history):
        self.flushed.append(list(messages))

    def _safe_print(self, *args, **kwargs):
        pass


def reasoning_only(reasoning):
    return SimpleNamespace(content=None, tool_calls=None, reasoning=reasoning, reasoning_details=None)


def fake_recover_empty_response(agent, assistant_message, response, finish_reason, **kw):
    return SimpleNamespace(
        action="break", result=None, final_response=EMPTY_PATH, turn_exit_reason="empty_response",
        active_system_prompt=kw["active_system_prompt"], api_call_count=kw["api_call_count"],
        preflight_compression_blocked=kw["preflight_compression_blocked"],
    )


def fake_apply_stop_gates(agent, final_msg, *, final_response, pending_verification_response,
                          pending_verification_response_previewed, **kw):
    return StopGateVerdict(
        continue_turn=False, final_response=final_response,
        pending_verification_response=pending_verification_response,
        pending_verification_response_previewed=pending_verification_response_previewed,
    )


def fake_apply_llm_output_transform(agent, final_response, **kw):
    return final_response, False, None


class FinishTextResponseTest(unittest.TestCase):
    def setUp(self):
        self._home = tempfile.TemporaryDirectory()
        self._old_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = self._home.name
        self._patches = [
            patch("agent.turn_final_response.recover_empty_response", side_effect=fake_recover_empty_response),
            patch("agent.turn_final_response.apply_stop_gates", side_effect=fake_apply_stop_gates),
            patch("agent.turn_finalizer.apply_llm_output_transform", side_effect=fake_apply_llm_output_transform),
        ]
        self.recover = self._patches[0].start()
        for p in self._patches[1:]:
            p.start()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        if self._old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self._old_home
        self._home.cleanup()

    def finish(self, reasoning, finish_reason):
        messages = [{"role": "user", "content": "Is the migration done?"}]
        return finish_text_response(
            FakeAgent(), assistant_message=reasoning_only(reasoning), response=None,
            finish_reason=finish_reason, messages=messages, api_messages=list(messages),
            conversation_history=[], api_call_count=1, user_message="Is the migration done?",
            active_system_prompt="system", final_response=None, _turn_exit_reason=None,
            _preflight_compression_blocked=False, codex_ack_continuations=0,
            truncated_response_parts=[], length_continue_retries=0,
            _pending_verification_response=None, _pending_verification_response_previewed=False,
            effective_task_id="t1",
        )

    def test_stop_with_cut_reasoning_is_not_the_final_answer(self):
        verdict = self.finish(CUT, "stop")
        self.assertNotEqual(verdict.final_response, CUT)
        self.assertEqual(verdict.final_response, EMPTY_PATH)
        self.recover.assert_called_once()

    def test_stop_with_whole_reasoning_is_the_final_answer(self):
        verdict = self.finish(ANSWER, "stop")
        self.assertEqual(verdict.action, "break")
        self.assertEqual(verdict.final_response, ANSWER)
        self.recover.assert_not_called()

    def test_length_with_whole_reasoning_is_not_the_final_answer(self):
        verdict = self.finish(ANSWER, "length")
        self.assertNotEqual(verdict.final_response, ANSWER)
        self.recover.assert_called_once()


if __name__ == "__main__":
    unittest.main()
