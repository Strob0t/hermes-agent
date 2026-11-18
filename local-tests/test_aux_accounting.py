"""Tests of the local aux-accounting patch (patch_aux_accounting.py), run in the image build.

The patch makes ``record_aux_usage`` price an auxiliary call with ``base_url=None`` so the lookup
stays offline. The test drives ``record_aux_usage`` with a published accounting context and asserts
that ``estimate_usage_cost`` is called with ``base_url=None`` even though a base URL was passed in,
and that the recorded usage reaches the session store.
"""
import types
import unittest
from unittest.mock import patch

import agent.aux_accounting as aux


class _Usage:
    def __init__(self):
        self.input_tokens = 10
        self.output_tokens = 5
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.reasoning_tokens = 0


class _Cost:
    amount_usd = 0.0001


def _response():
    return types.SimpleNamespace(model="claude-3-5-haiku", usage=_Usage())


class AuxBaseUrlTest(unittest.TestCase):
    def setUp(self):
        self.recorded = {}
        session_db = types.SimpleNamespace(record_auxiliary_usage=lambda *a, **k: None)
        self.token = aux.set_accounting_context(session_db=session_db, session_id="s1")
        self.addCleanup(aux.reset_accounting_context, self.token)

    def _run(self, base_url):
        captured = {}

        def fake_estimate(model, usage, *, provider=None, base_url=None):
            captured["base_url"] = base_url
            captured["provider"] = provider
            return _Cost()

        def fake_record(func, *args, **kwargs):
            # call_off_event_loop(session_db.record_auxiliary_usage, session_id, task, **fields)
            self.recorded["args"] = args
            self.recorded["kwargs"] = kwargs

        with patch("agent.usage_pricing.estimate_usage_cost", fake_estimate), \
                patch("agent.usage_pricing.normalize_usage", lambda raw, provider=None: raw), \
                patch("agent.usage_pricing.with_served_service_tier", lambda usage, response: usage), \
                patch("agent.runtime_session_store.call_off_event_loop", fake_record):
            aux.record_aux_usage(_response(), task="vision", provider="anthropic", base_url=base_url)
        return captured

    def test_base_url_is_forced_to_none(self):
        captured = self._run(base_url="http://bifrost:8080/v1")
        self.assertIsNone(captured["base_url"])
        self.assertEqual(captured["provider"], "anthropic")

    def test_base_url_is_none_even_without_an_input_base_url(self):
        captured = self._run(base_url=None)
        self.assertIsNone(captured["base_url"])

    def test_usage_is_recorded_with_the_estimated_cost(self):
        self._run(base_url="http://bifrost:8080/v1")
        kwargs = self.recorded["kwargs"]
        self.assertEqual(kwargs["input_tokens"], 10)
        self.assertEqual(kwargs["output_tokens"], 5)
        self.assertEqual(kwargs["estimated_cost_usd"], 0.0001)
        # the originally-resolved billing base_url is still recorded for the ledger
        self.assertEqual(kwargs["billing_base_url"], "http://bifrost:8080/v1")

    def test_excluded_task_records_nothing(self):
        self._run_excluded = self.recorded
        with patch("agent.usage_pricing.estimate_usage_cost", lambda *a, **k: _Cost()):
            aux.record_aux_usage(_response(), task="moa_aggregator", provider="anthropic",
                                 base_url="http://bifrost:8080/v1")
        self.assertEqual(self.recorded, {})


if __name__ == "__main__":
    unittest.main()
