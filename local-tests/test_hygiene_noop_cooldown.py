"""Tests of the no-op hygiene cooldown port (finding #20), run in the image build.

Session hygiene recorded a failure cooldown only when the compressor aborted. A compression that ran
but did not reduce the context (no rotation, no in-place compaction, or no material shrink) left no
cooldown, so the next turn ran the same useless compression again. The patch adds the predicate
``gateway.run.hygiene_noop_should_cool`` and ``_hmwa_hygiene_apply_result`` records the escalating
failure cooldown for that no-op case. The tests use a real store in a temp HERMES_HOME and show:
 - a no-op compression result sets the cooldown, and the next hygiene check skips compression;
 - an effective (in-place, shrinking) compression sets no cooldown;
 - an aborted compression keeps the old behavior: cooldown with the summary error.
"""
import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gateway.run_turn_hygiene import GatewayTurnHygieneMixin
from persistence.conversation.sessions import NewSession
from persistence_assembly import open_persistence, profile_scope

SID = "sess-hyg-noop"
KEY = "agent:main:telegram:dm:1"


def _history(n=40):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 400} for i in range(n)]


class _Host(GatewayTurnHygieneMixin):
    """Gateway stand-in with the real hygiene mixin and a real store; adapters are absent."""

    def __init__(self, store):
        self.persistence = store

    def _session_state(self, _key):
        raise KeyError(_key)

    def _peek_session_state(self, _key):
        return None

    def _delivery_adapter_for(self, _source):
        return None

    async def _session_has_compression_in_flight(self, _key):
        return False


def _settings():
    return SimpleNamespace(
        failure_cooldown_seconds=300.0, threshold_pct=0.5, model="fixture-model", base_url="", api_key="",
        config_context_length=1000, provider="", hard_msg_limit=5000)


def _attempt(*, aborted=False, in_place=False, error=None):
    comp = SimpleNamespace(_last_compress_aborted=aborted, _last_summary_error=error,
                           _last_aux_model_failure_model=None)
    agent = SimpleNamespace(session_id=SID, _last_compaction_in_place=in_place, context_compressor=comp)
    return SimpleNamespace(agent=agent, commit_fence=SimpleNamespace(is_cancelled=False), history=None, meta={})


class HygieneNoopCooldownTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        home = Path(self._tmp.name)
        self._env = patch.dict(os.environ, {"HERMES_HOME": str(home), "HERMES_MANAGED_DIR": str(home / "managed")})
        self._env.start()
        self.store = open_persistence(profile_scope(home=home))
        self.store.sessions.create(NewSession(session_id=SID, source="telegram"))
        self.host = _Host(self.store)
        self.hs = _settings()

    def tearDown(self):
        self.store.close()
        self._env.stop()
        self._tmp.cleanup()

    def _entry(self):
        return SimpleNamespace(session_id=SID, last_prompt_tokens=2000)

    def _apply(self, attempt, compressed, history):
        plan = self.host._HygienePlan(True, 2000, len(history), 950)
        asyncio.run(self.host._hmwa_hygiene_apply_result(
            attempt, self.hs, compressed, history, plan, session_entry=self._entry(), session_key=KEY,
            source=SimpleNamespace(chat_id=None, platform=None), _quick_key="qk", run_generation=0))

    def _cooldown(self):
        return self.store.messages.read_compression_cooldown(SID, time.time())

    def test_noop_compression_sets_cooldown_and_next_check_skips(self):
        history = _history()
        before = asyncio.run(self.host._hmwa_hygiene_plan(self.hs, history, self._entry(), KEY))
        self.assertTrue(before.needs_compress)

        self._apply(_attempt(), list(history), history)

        cooldown = self._cooldown()
        self.assertIsNotNone(cooldown)
        self.assertGreater(cooldown.until - time.time(), 0)
        after = asyncio.run(self.host._hmwa_hygiene_plan(self.hs, history, self._entry(), KEY))
        self.assertFalse(after.needs_compress)

    def test_effective_compression_sets_no_cooldown(self):
        history = _history()
        compressed = [{"role": "user", "content": "summary"}] + history[-2:]

        self._apply(_attempt(in_place=True), compressed, history)

        self.assertIsNone(self._cooldown())

    def test_aborted_compression_keeps_cooldown_with_summary_error(self):
        history = _history()

        self._apply(_attempt(aborted=True, error="fixture summary failure"), list(history), history)

        cooldown = self._cooldown()
        self.assertIsNotNone(cooldown)
        self.assertEqual(cooldown.error, "fixture summary failure")


if __name__ == "__main__":
    unittest.main()
