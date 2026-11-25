"""Tests of the truncated-reasoning promotion gate (port of fork fix/no-truncated-reasoning-promotion).

The patch moves the reasoning-only promotion decision of the final-text turn into
``agent.reasoning_promotion.promoted_final_text`` and adds a shape check
(``looks_like_truncated_promoted_reasoning``): a reasoning field that looks cut off is not
promoted to the final answer. The tests show:
 - a reasoning-only response with finish_reason ``length`` is not promoted;
 - the same response with finish_reason ``stop`` is promoted as before;
 - a ``stop`` response whose reasoning ends mid-token is not promoted;
 - complete short answers do not look truncated, cut tails do.
"""
import os
import tempfile
import unittest
from types import SimpleNamespace

from agent.reasoning_promotion import looks_like_truncated_promoted_reasoning, promoted_final_text

# Made at import: an earlier test in the same run can point TMPDIR at a removed folder.
_HOME = tempfile.TemporaryDirectory()

ANSWER = "The migration is complete. All 12 tables were copied and the checksums match."


class FakeAgent:
    """Agent on the trusted local Nemotron parser route, so promotion is permitted."""

    model = "nemotron-3.5-lightning"
    provider = "custom"
    requested_provider = ""
    base_url = "http://127.0.0.1:8000/v1"
    api_mode = "chat_completions"
    _custom_providers = []

    def _extract_reasoning(self, message):
        return message.reasoning


def reasoning_only(reasoning):
    return SimpleNamespace(content=None, tool_calls=None, reasoning=reasoning, reasoning_details=None)


class PromotionTest(unittest.TestCase):
    def setUp(self):
        self._old_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = _HOME.name

    def tearDown(self):
        if self._old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self._old_home

    def promote(self, reasoning, finish_reason):
        return promoted_final_text(FakeAgent(), reasoning_only(reasoning), finish_reason, 1, [])

    def test_length_cut_reasoning_is_not_promoted(self):
        self.assertIsNone(self.promote(ANSWER, "length"))
        self.assertIsNone(self.promote("The fix was already applied on 2026-", "length"))

    def test_clean_stop_reasoning_is_promoted(self):
        self.assertEqual(self.promote(ANSWER, "stop"), ANSWER)

    def test_clean_stop_with_cut_reasoning_is_not_promoted(self):
        self.assertIsNone(self.promote("The fix was already applied on 2026-", "stop"))


class TruncationShapeTest(unittest.TestCase):
    def test_complete_answers_do_not_look_truncated(self):
        for text in ("Done.", "42", "SQLite", "[SILENT]", "Here's what the flag is for", ANSWER):
            self.assertFalse(looks_like_truncated_promoted_reasoning(text), text)

    def test_cut_tails_look_truncated(self):
        for text in ("applied on 2026-", "see the list (first", "the result is the", "```py\nx = 1"):
            self.assertTrue(looks_like_truncated_promoted_reasoning(text), text)


if __name__ == "__main__":
    unittest.main()
