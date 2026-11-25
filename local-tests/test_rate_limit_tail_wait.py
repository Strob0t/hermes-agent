"""Tests of the opt-in rate-limit tail wait (port of finding #7), run in the image build.

When the rebuilt-restart limit ends a turn whose fallback chain failed, and a rate-limit cooldown
(``agent._rate_limited_until``) is armed, ``apply_retry_restarts`` can wait the cooldown out once,
restore the primary runtime and re-issue the iteration. Config key
``agent.rate_limit_tail_wait_seconds`` (default 0 = off, clamped to at most 60) sets the longest
cooldown that is waited for. The ``_LoopState._rate_limit_tail_wait_used`` flag allows one grant
per turn. The tests show:
 - key 0: no wait, the turn ends with ``rebuilt_restart_limit_exceeded`` as before;
 - key 30, cooldown 10 s: exactly 1 wait, then 1 retry; a second rate limit ends the turn;
 - key 30, cooldown 90 s: no wait, the turn ends;
 - key 500 is clamped to 60;
 - an interrupt during the wait ends it within 1 s and ends the turn.
A fake clock replaces ``time`` in the module, so no test sleeps for real (except the short
interrupt test, which waits at most a fraction of a second).
"""
import dataclasses
import os
import tempfile
import threading
import time as real_time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import agent.turn_iteration_prep as tip
from agent.conversation_loop import _LoopState, _run_phase
from agent.turn_iteration_prep import apply_retry_restarts

MAX_RETRIES = 3


class FakeClock:
    """Monotonic clock that only moves when ``sleep`` is called; records each sleep."""

    def __init__(self, start=1000.0):
        self.now = start
        self.sleeps = []

    def monotonic(self):
        return self.now

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeBudget:
    def __init__(self):
        self.refunds = 0

    def refund(self):
        self.refunds += 1


class FakeAgent:
    """Minimal agent for the rebuilt-restart branch: a cooldown slot, an interrupt flag, counters."""

    def __init__(self, rate_limited_until=0.0):
        self._rate_limited_until = rate_limited_until
        self._interrupt_requested = False
        self._tool_interrupt_reason = None
        self.iteration_budget = FakeBudget()
        self.restored = 0
        self.statuses = []

    def _restore_primary_runtime(self):
        self.restored += 1
        return True

    def _emit_diagnostic_status(self, message):
        self.statuses.append(message)

    def _touch_activity(self, *_a, **_k):
        pass


def _retry_state():
    return SimpleNamespace(
        restart_with_redirected_messages=False, restart_with_compressed_messages=False,
        restart_with_rebuilt_messages=True, restart_with_length_continuation=False,
    )


def _phase_state(**over):
    """Loop state at the rebuilt-restart limit, with the turn default of the one-grant flag."""
    flag = next(f for f in dataclasses.fields(_LoopState) if f.name == "_rate_limit_tail_wait_used")
    values = dict(
        _retry=_retry_state(), response=None, interrupted=False, messages=[],
        conversation_history=[], user_message="hi", api_kwargs={}, current_turn_user_idx=0,
        final_response=None, retry_count=0, max_retries=MAX_RETRIES, api_call_count=5,
        restart_count=MAX_RETRIES, length_continue_retries=0,
        _preflight_compression_blocked=True, _turn_exit_reason="unknown",
        _rate_limit_tail_wait_used=flag.default,
    )
    values.update(over)
    return SimpleNamespace(**values)


class _ConfigCase(unittest.TestCase):
    """Each test runs with its own temp HERMES_HOME holding ``agent.rate_limit_tail_wait_seconds``."""

    key = None

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        if self.key is not None:
            with open(os.path.join(self._tmp.name, "config.yaml"), "w", encoding="utf-8") as fh:
                fh.write(f"agent:\n  rate_limit_tail_wait_seconds: {self.key}\n")
        env = patch.dict(os.environ, {"HERMES_HOME": self._tmp.name})
        env.start()
        self.addCleanup(env.stop)
        self.clock = FakeClock()
        clock_patch = patch.object(tip, "time", self.clock, create=True)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)

    def run_limit(self, agent, state):
        return _run_phase(apply_retry_restarts, agent, state)


class KeyOffTest(_ConfigCase):
    key = 0

    def test_default_off_ends_turn_without_wait(self):
        agent = FakeAgent(rate_limited_until=self.clock.now + 10)
        state = _phase_state()
        verdict = self.run_limit(agent, state)
        self.assertEqual(verdict.action, "break")
        self.assertEqual(state._turn_exit_reason, "rebuilt_restart_limit_exceeded")
        self.assertEqual(self.clock.sleeps, [])
        self.assertEqual((agent.restored, agent.iteration_budget.refunds), (0, 0))


class ShortCooldownTest(_ConfigCase):
    key = 30

    def test_waits_once_then_retries_once(self):
        agent = FakeAgent(rate_limited_until=self.clock.now + 10)
        state = _phase_state()
        verdict = self.run_limit(agent, state)
        self.assertEqual(verdict.action, "continue")
        self.assertAlmostEqual(sum(self.clock.sleeps), 10.0, places=6)
        self.assertTrue(all(s <= 1.0 for s in self.clock.sleeps))
        self.assertEqual((agent.restored, agent.iteration_budget.refunds), (1, 1))
        self.assertEqual(state.api_call_count, 4)
        self.assertFalse(state._retry.restart_with_rebuilt_messages)
        self.assertFalse(state._preflight_compression_blocked)
        self.assertTrue(state._rate_limit_tail_wait_used)

    def test_second_rate_limit_in_same_turn_ends_turn(self):
        agent = FakeAgent(rate_limited_until=self.clock.now + 10)
        state = _phase_state()
        self.assertEqual(self.run_limit(agent, state).action, "continue")
        slept = list(self.clock.sleeps)
        agent._rate_limited_until = self.clock.now + 10
        state._retry.restart_with_rebuilt_messages = True
        verdict = self.run_limit(agent, state)
        self.assertEqual(verdict.action, "break")
        self.assertEqual(state._turn_exit_reason, "rebuilt_restart_limit_exceeded")
        self.assertEqual(self.clock.sleeps, slept)
        self.assertEqual(agent.restored, 1)


class LongCooldownTest(_ConfigCase):
    key = 30

    def test_cooldown_above_key_ends_turn_without_wait(self):
        agent = FakeAgent(rate_limited_until=self.clock.now + 90)
        state = _phase_state()
        verdict = self.run_limit(agent, state)
        self.assertEqual(verdict.action, "break")
        self.assertEqual(state._turn_exit_reason, "rebuilt_restart_limit_exceeded")
        self.assertEqual(self.clock.sleeps, [])
        self.assertFalse(state._rate_limit_tail_wait_used)


class ClampTest(_ConfigCase):
    key = 500

    def test_key_is_clamped_to_60(self):
        self.assertEqual(tip.rate_limit_tail_wait_seconds(), 60.0)

    def test_cooldown_between_60_and_key_is_not_waited(self):
        agent = FakeAgent(rate_limited_until=self.clock.now + 61)
        verdict = self.run_limit(agent, _phase_state())
        self.assertEqual(verdict.action, "break")
        self.assertEqual(self.clock.sleeps, [])
        agent = FakeAgent(rate_limited_until=self.clock.now + 60)
        self.assertEqual(self.run_limit(agent, _phase_state()).action, "continue")


class InterruptTest(unittest.TestCase):
    """Real clock, 30 s cooldown; an interrupt 0.1 s into the wait must end it within 1 s."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        with open(os.path.join(self._tmp.name, "config.yaml"), "w", encoding="utf-8") as fh:
            fh.write("agent:\n  rate_limit_tail_wait_seconds: 30\n")
        env = patch.dict(os.environ, {"HERMES_HOME": self._tmp.name})
        env.start()
        self.addCleanup(env.stop)

    def test_interrupt_ends_wait_within_one_second(self):
        agent = FakeAgent(rate_limited_until=real_time.monotonic() + 30)
        timer = threading.Timer(0.1, lambda: setattr(agent, "_interrupt_requested", True))
        state = _phase_state()
        started = real_time.monotonic()
        timer.start()
        try:
            verdict = _run_phase(apply_retry_restarts, agent, state)
        finally:
            timer.cancel()
        elapsed = real_time.monotonic() - started
        self.assertLess(elapsed, 1.1)
        self.assertEqual(verdict.action, "break")
        self.assertTrue(state._turn_exit_reason.startswith("interrupted_during_api_call"))
        self.assertEqual((agent.restored, agent.iteration_budget.refunds), (0, 0))


if __name__ == "__main__":
    unittest.main()
