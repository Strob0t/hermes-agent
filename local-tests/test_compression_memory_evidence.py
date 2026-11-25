"""Tests of the compression memory-evidence port (agent/context_compressor_evidence.py).

A memory provider (here openviking) injects its recall into the user turn: the fork appends a
``<memory-context>`` block (``build_memory_context_block``) to the row's ``api_content`` sidecar,
while ``content`` keeps only the typed text. Before the patch the summarizer read only ``content``,
so compaction dropped the recall facts. ``SummaryEvidenceMixin`` now owns
``_serialize_records_for_summary`` in ``ContextCompressor`` and adds the appended sidecar as a
separately labelled, bounded evidence record; the deterministic fallback summary keeps it too.
The tests show:
 - after compaction, the summary input (and the fallback summary) contains the recall facts;
 - a very large recall block is cut to a bound, and the typed ask stays ahead of the evidence;
 - a transcript without memory serializes exactly as before, and a rewritten sidecar is not evidence.
"""
import copy
import os
import tempfile
import unittest
from unittest.mock import patch

_HOME = tempfile.mkdtemp(prefix="hermes-home-")
_ENV = patch.dict(os.environ, {"HERMES_HOME": _HOME})


def setUpModule():
    # The agent imports touch HERMES_HOME at import time, so they run after the patch starts.
    global COMPRESSED_SUMMARY_METADATA_KEY, ContextCompressor, compose_user_api_content
    _ENV.start()
    from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY, ContextCompressor
    from agent.turn_context import compose_user_api_content


def tearDownModule():
    _ENV.stop()


FACT = "Quartz orchard appointment: 17 November 2031, 14:37 CET."
RECALL = f"## OpenViking recall\n- {FACT}\n- prefers draft PRs"
ASK = "What is my Quartz orchard appointment?"


def _recall_row(recall=RECALL, ask=ASK):
    row = {"role": "user", "content": ask}
    row["api_content"] = compose_user_api_content(ask, recall, "")
    return row


def _history(first):
    messages = [first, {"role": "assistant", "content": "I found the appointment."}]
    for i in range(12):
        messages.extend([
            {"role": "user", "content": f"Step {i}: " + "x" * 30},
            {"role": "assistant", "content": f"Completed step {i}. " + "y" * 30},
        ])
    messages.append({"role": "user", "content": "And when exactly?"})
    return messages


def _compressor():
    c = ContextCompressor(model="test/model", config_context_length=100_000,
                          protect_first_n=0, protect_last_n=2, quiet_mode=True)
    c.tail_token_budget = 10
    return c


def _compress(messages, summary_available):
    prompts = []

    def summarize(self, prompt_messages, started_at):
        text = "\n".join(str(m.get("content", "")) for m in prompt_messages)
        prompts.append(text)
        if not summary_available:
            raise RuntimeError("summary provider unavailable")
        return "## Critical Context\n" + (FACT if FACT in text else "No appointment date supplied.")

    original = ContextCompressor._call_summary_llm
    ContextCompressor._call_summary_llm = summarize
    try:
        compressor = _compressor()
        compressed = compressor.compress(messages, force=True)
    finally:
        ContextCompressor._call_summary_llm = original
    summaries = [r for r in compressed if r.get(COMPRESSED_SUMMARY_METADATA_KEY)]
    return prompts, summaries, compressor


class RecallReachesSummaryTest(unittest.TestCase):
    def test_wrapper_shape_is_the_real_memory_fence(self):
        sidecar = _recall_row()["api_content"]
        self.assertTrue(sidecar.startswith(ASK + "\n\n<memory-context>\n"))
        self.assertIn(FACT, sidecar)

    def test_summary_input_contains_recall_facts(self):
        messages = _history(_recall_row())
        before = copy.deepcopy(messages)
        prompts, summaries, _ = _compress(messages, summary_available=True)
        self.assertTrue(prompts)
        self.assertIn(FACT, prompts[0])
        self.assertIn(ASK, prompts[0])
        self.assertTrue(summaries)
        self.assertIn(FACT, summaries[0]["content"])
        self.assertEqual(messages, before)

    def test_fallback_summary_keeps_recall_facts(self):
        prompts, summaries, compressor = _compress(_history(_recall_row()), summary_available=False)
        self.assertTrue(compressor._last_summary_fallback_used)
        self.assertTrue(summaries)
        self.assertIn(FACT, summaries[0]["content"])


class EvidenceBoundTest(unittest.TestCase):
    def test_large_recall_is_bounded_and_ask_comes_first(self):
        compressor = _compressor()
        huge = RECALL + "\n" + "\n".join(f"- filler fact {i} " + "z" * 80 for i in range(2_000))
        row = _recall_row(recall=huge)
        self.assertGreater(len(row["api_content"]), 100_000)
        record = compressor._serialize_records_for_summary([row])[0]
        self.assertIn(FACT, record)
        self.assertLessEqual(len(record), compressor._CONTENT_MAX + len("[USER]: "))
        self.assertLess(record.index(ASK), record.index(FACT))

    def test_fallback_evidence_stays_inside_the_fallback_cap(self):
        huge = RECALL + "\n" + "z" * 200_000
        _, summaries, _ = _compress(_history(_recall_row(recall=huge)), summary_available=False)
        self.assertIn(FACT, summaries[0]["content"])
        self.assertLess(len(summaries[0]["content"]), 20_000)


class NoMemoryUnchangedTest(unittest.TestCase):
    def test_transcript_without_memory_serializes_as_before(self):
        turns = [
            {"role": "user", "content": ASK},
            {"role": "assistant", "content": "I found the appointment."},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
        records = _compressor()._serialize_records_for_summary(copy.deepcopy(turns))
        self.assertEqual(records, [f"[USER]: {ASK}", "[ASSISTANT]: I found the appointment.", "[TOOL RESULT c1]: ok"])

    def test_rewritten_sidecar_is_not_evidence(self):
        row = {"role": "user", "content": ASK, "api_content": "a different wire text " + FACT}
        record = _compressor()._serialize_records_for_summary([row])[0]
        self.assertEqual(record, f"[USER]: {ASK}")


if __name__ == "__main__":
    unittest.main()
