"""Tests of the local compression-anchor patch (compression-anchor.patch), run in the image build.

An API client turn is persisted with display_kind internal_notification (session_ingress marks every API
turn internal); the patch tags it api_inbound so compaction anchors on it, and a request with a finished
answer is never restated as in progress.
"""
import types
import unittest
from unittest.mock import MagicMock, patch

from agent.context_compressor import _INFLIGHT_TASK_REPLAY_HEADER, ContextCompressor

OLD_ASK = "OLD_ASK_check_the_first_service"
OLD_ANSWER = "The first service is checked."
LIVE_ASK = "LIVE_ASK_now_replace_the_second_service"
NOTICE = "Background job finished."
JOB = "JOB_PROMPT_write_a_digest"


def live(text=LIVE_ASK):
    return {"role": "user", "content": text, "display_kind": "internal_notification",
            "display_metadata": {"gateway_input_owner": "owner", "api_inbound": True}}


def notice(text=NOTICE):
    return {"role": "user", "content": text, "display_kind": "internal_notification",
            "display_metadata": {"gateway_input_owner": "owner"}}


def tools(count, start=0):
    rows = []
    for i in range(start, start + count):
        rows.append({"role": "assistant", "content": f"step {i}",
                     "tool_calls": [{"id": f"c{i}", "function": {"name": "terminal", "arguments": "{}"}}]})
        rows.append({"role": "tool", "tool_call_id": f"c{i}", "content": ("tool output " * 200) + str(i)})
    return rows


def api_transcript():
    """Today's shape: head, an answered earlier request, the live API turn, a long tool tail."""
    return [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": OLD_ASK},
        *tools(3, 100),
        {"role": "assistant", "content": OLD_ANSWER},
        live(),
        *tools(40),
    ]


def compress(messages):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "## Historical Task Snapshot\nUser asked: earlier work\n## Summary\nSteps ran."
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)
    compressor.tail_token_budget = 500
    with patch("agent.context_compressor.call_llm", return_value=response):
        return compressor.compress(messages, current_tokens=200_000, force=True)


def text(row):
    content = row.get("content")
    return content if isinstance(content, str) else str(content)


def restated(rows, marker):
    """Rows whose text after the in-flight header names marker (the restated active request)."""
    return [r for r in rows if marker in text(r).rpartition(_INFLIGHT_TASK_REPLAY_HEADER)[2]
            and _INFLIGHT_TASK_REPLAY_HEADER in text(r)]


class ActionableTest(unittest.TestCase):
    def test_api_inbound_turn_is_actionable(self):
        self.assertTrue(ContextCompressor._is_actionable_user_turn(live()))

    def test_internal_notification_stays_not_actionable(self):
        self.assertFalse(ContextCompressor._is_actionable_user_turn(notice()))

    def test_marker_needs_the_internal_kind(self):
        row = live()
        row["display_kind"] = "hidden"
        self.assertFalse(ContextCompressor._is_actionable_user_turn(row))


class InflightTest(unittest.TestCase):
    def test_live_api_turn_is_the_inflight_task(self):
        rows = api_transcript()
        self.assertIs(ContextCompressor._find_inflight_user_task(rows), next(r for r in rows if text(r) == LIVE_ASK))

    def test_answered_request_before_a_notice_is_not_inflight(self):
        rows = [{"role": "user", "content": OLD_ASK}, {"role": "assistant", "content": OLD_ANSWER}, notice(), *tools(2)]
        self.assertIsNone(ContextCompressor._find_inflight_user_task(rows))

    def test_provisional_answer_keeps_the_task_inflight(self):
        ask = {"role": "user", "content": OLD_ASK}
        rows = [ask, {"role": "assistant", "content": "draft", "finish_reason": "verification_required"}, notice(),
                *tools(2)]
        self.assertIs(ContextCompressor._find_inflight_user_task(rows), ask)

    def test_unanswered_request_stays_inflight(self):
        ask = {"role": "user", "content": OLD_ASK}
        self.assertIs(ContextCompressor._find_inflight_user_task([ask, *tools(2)]), ask)

    def test_finished_turn_is_not_inflight(self):
        rows = [{"role": "user", "content": OLD_ASK}, *tools(2), {"role": "assistant", "content": OLD_ANSWER}]
        self.assertIsNone(ContextCompressor._find_inflight_user_task(rows))


class CompressTest(unittest.TestCase):
    def test_old_request_is_not_restated_and_live_turn_survives(self):
        out = compress(api_transcript())
        self.assertEqual(restated(out, OLD_ASK), [])
        last_user = [r for r in out if r.get("role") == "user"][-1]
        self.assertTrue(restated(out, LIVE_ASK) or text(last_user) == LIVE_ASK)

    def test_genuine_inflight_job_prompt_is_still_restated(self):
        rows = [{"role": "system", "content": "You are Hermes."}, {"role": "user", "content": JOB}, *tools(40)]
        out = compress(rows)
        self.assertTrue(restated(out, JOB))

    def test_first_api_turn_in_the_head_is_restated(self):
        rows = [{"role": "system", "content": "You are Hermes."}, live(), *tools(40)]
        out = compress(rows)
        self.assertTrue(restated(out, LIVE_ASK))


class MetadataTest(unittest.TestCase):
    def test_api_inbound_flag_reaches_the_user_row_metadata(self):
        from gateway.response_filters import persisted_user_metadata_for_event
        event = types.SimpleNamespace(internal=True, metadata={"api_inbound": True}, message_id="m1")
        self.assertEqual(persisted_user_metadata_for_event(event).get("api_inbound"), True)

    def test_other_events_get_no_flag(self):
        from gateway.response_filters import persisted_user_metadata_for_event
        event = types.SimpleNamespace(internal=True, metadata={}, message_id="m1")
        self.assertNotIn("api_inbound", persisted_user_metadata_for_event(event))


if __name__ == "__main__":
    unittest.main()
