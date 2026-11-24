"""Tests of the runaway-loop patch (#107) in agent/repetition_guard.py, run in the image build.

The patch lets ``is_runaway_repetition`` cut a loop that never emits a newline. When one line
holds at least half of the text and is an exact periodic run, the distinct lines written before
the loop no longer veto the cut. The tests show:
 - distinct lines followed by a long newline-free periodic run -> True;
 - the same distinct lines without the loop -> False;
 - normal prose with one long line that is not periodic -> False.
"""
import hashlib
import unittest

from agent.repetition_guard import is_repetition_dominated, is_runaway_repetition

_DISTINCT_LINES = "".join(
    f"Check {i}: service {i} answers on port {8000 + i}, its config parses and the redirect for host-{i} is in place.\n"
    for i in range(16)
)


def _prose_line(length: int) -> str:
    words = []
    total = 0
    i = 0
    while total < length:
        word = hashlib.sha256(str(i).encode()).hexdigest()[:6]
        words.append(word)
        total += len(word) + 1
        i += 1
    return " ".join(words)[:length]


class TestNewlineFreeLoop(unittest.TestCase):
    def test_distinct_lines_then_newline_free_loop_is_runaway(self):
        loop_line = " hmm —" * 890 + " a" + "aaaaaaaa" * 5000
        text = _DISTINCT_LINES + loop_line

        self.assertGreater(len(text.splitlines()), 5)
        self.assertTrue(is_runaway_repetition(text))

    def test_same_distinct_lines_without_loop_are_not_runaway(self):
        self.assertFalse(is_repetition_dominated(_DISTINCT_LINES))
        self.assertFalse(is_runaway_repetition(_DISTINCT_LINES))

    def test_normal_prose_with_one_long_non_periodic_line_is_not_runaway(self):
        text = _DISTINCT_LINES + _prose_line(6000) + "\n"

        self.assertFalse(is_runaway_repetition(text))


if __name__ == "__main__":
    unittest.main()
