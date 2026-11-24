"""Tests of the local compaction-complete-coverage port (finding 45b771154022), run in the image build.

Summarizer input above ``_SUMMARY_INPUT_MAX_CHARS`` was sampled with an omitted middle. The patch
splits the serialized turns into ordered passes below the bound, keeps an assistant tool-call row in
the same pass as its tool results, and folds each next pass into the running summary. Only the final
pass commits ``_previous_summary``. The tests show:
 - every numbered marker of an oversized transcript reaches the summarizer (union of all calls);
 - each tool call and its result travel in the same summarizer call;
 - a failure in pass 2 keeps ``_previous_summary`` and the transcript unchanged;
 - input below the bound gives exactly 1 summarizer call.
"""
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

_HOME = tempfile.mkdtemp(prefix="hermes-home-")
os.environ["HERMES_HOME"] = _HOME

from agent.context_compressor import ContextCompressor  # noqa: E402

PRIOR = "## Summary\nPRIOR_SUMMARY_FROM_EARLIER_COMPACTION"
GROUPS = 80
FILLER = "x" * 3800


def tearDownModule():
    shutil.rmtree(_HOME, ignore_errors=True)


def marker(i):
    return f"MARKER-{i:04d}"


def transcript(groups=GROUPS, filler=FILLER):
    rows = [{"role": "user", "content": "please inspect every service"}]
    for i in range(groups):
        rows.append({"role": "assistant", "content": f"step {i}",
                     "tool_calls": [{"id": f"call{i}", "type": "function",
                                     "function": {"name": "terminal", "arguments": "{}"}}]})
        rows.append({"role": "tool", "tool_call_id": f"call{i}", "content": f"{marker(i)} {filler}"})
    return rows


def response(text="## Historical Task Snapshot\nUser asked: inspect services\n## Summary\nDone."):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.choices[0].finish_reason = "stop"
    return resp


class Recorder:
    """Mock summarizer: records each request; raises on the call numbers in ``fail_on``."""

    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs.get("messages") or [])
        if len(self.calls) in self.fail_on:
            raise ValueError("summarizer pass failed")
        return response()

    def text(self, index):
        return "\n".join(str(m.get("content") or "") for m in self.calls[index])


def compressor(**kwargs):
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        comp = ContextCompressor(model="test", quiet_mode=True, **kwargs)
    return comp


def summarize(comp, turns, recorder):
    with patch("agent.context_compressor.call_llm", side_effect=recorder):
        return comp._generate_summary(turns)


class CompleteCoverageTest(unittest.TestCase):
    def test_every_marker_reaches_the_summarizer(self):
        turns = transcript()
        self.assertGreater(sum(len(str(t.get("content"))) for t in turns), 160_000)
        recorder = Recorder()
        summary = summarize(compressor(), turns, recorder)
        self.assertTrue(summary)
        self.assertGreater(len(recorder.calls), 1)
        seen = "\n".join(recorder.text(i) for i in range(len(recorder.calls)))
        missing = [marker(i) for i in range(GROUPS) if marker(i) not in seen]
        self.assertEqual(missing, [])

    def test_tool_call_and_result_share_a_call(self):
        recorder = Recorder()
        summarize(compressor(), transcript(), recorder)
        for i in range(GROUPS):
            holders = [n for n, msgs in enumerate(recorder.calls)
                       if any(m.get("role") == "tool" and m.get("tool_call_id") == f"call{i}"
                              and marker(i) in str(m.get("content")) for m in msgs)]
            self.assertEqual(len(holders), 1, f"result of call{i} in calls {holders}")
            call_msgs = recorder.calls[holders[0]]
            self.assertTrue(any(m.get("role") == "assistant" and any(
                tc.get("id") == f"call{i}" for tc in m.get("tool_calls") or ()) for m in call_msgs))


class LaterPassFailureTest(unittest.TestCase):
    def test_failure_in_pass_two_keeps_previous_summary(self):
        comp = compressor()
        comp._previous_summary = PRIOR
        turns = transcript()
        before = [dict(t) for t in turns]
        recorder = Recorder(fail_on={2})
        self.assertIsNone(summarize(comp, turns, recorder))
        self.assertEqual(len(recorder.calls), 2)
        self.assertEqual(comp._previous_summary, PRIOR)
        self.assertEqual(turns, before)

    def test_failure_in_pass_two_leaves_messages_unchanged(self):
        comp = compressor(protect_first_n=1, protect_last_n=2, abort_on_summary_failure=True)
        comp._previous_summary = PRIOR
        messages = [{"role": "system", "content": "system prompt"}] + transcript() + [
            {"role": "user", "content": "latest ask"}, {"role": "assistant", "content": "latest answer"}]
        before = [dict(m) for m in messages]
        recorder = Recorder(fail_on={2})
        with patch("agent.context_compressor.call_llm", side_effect=recorder):
            result = comp.compress(messages, current_tokens=200_000, force=True)
        self.assertEqual(len(recorder.calls), 2)
        self.assertEqual(result, before)
        self.assertEqual(comp._previous_summary, PRIOR)


class BelowBoundTest(unittest.TestCase):
    def test_small_input_is_one_call(self):
        turns = transcript(groups=5, filler="y" * 200)
        recorder = Recorder()
        summary = summarize(compressor(), turns, recorder)
        self.assertTrue(summary)
        self.assertEqual(len(recorder.calls), 1)
        self.assertTrue(all(marker(i) in recorder.text(0) for i in range(5)))


if __name__ == "__main__":
    unittest.main()
