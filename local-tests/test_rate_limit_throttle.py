"""Tests of the opt-in client-side RPM/TPM throttle (agent/rate_limit_throttle.py), run in the image build.

The patch adds a per-(provider, model) request pacer (``providers.<id>.requests_per_minute``) and an
input-token bucket (``providers.<id>.input_tokens_per_minute``), each with a per-model override under
``providers.<id>.models.<model>``. The hooks are ``perform_api_call``, ``record_response_usage``, the
iteration-summary call and the CLI status bar. Missing or <= 0 is off (base behavior). The ceilings
are resolved once per agent, not on every API call. The tests show:
 - with no keys, a request never waits;
 - with ``requests_per_minute: 60``, the second of 2 simultaneous calls waits about 1 s (fake clock);
 - a per-model value wins over the provider value;
 - ``input_tokens_per_minute`` delays the next request by the recorded usage, and the bucket drains;
 - a value <= 0 is off;
 - repeated calls read the config only once per agent;
 - the 2 keys are accepted provider keys (no "unknown config keys" warning).
"""
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import hermes_yaml as yaml

import agent.rate_limit_throttle as rlt
import hermes_cli.config as hconfig
from hermes_cli.config_providers import _normalize_custom_provider_entry


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)


class FakeAgent:
    log_prefix = ""

    def __init__(self, provider="acme", model="m1"):
        self.provider = provider
        self.model = model
        self.printed = []

    def _vprint(self, text, force=False):
        self.printed.append(text)


class ThrottleTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = self._tmp.name
        env = patch.dict(os.environ, {"HERMES_HOME": self.home})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(self._tmp.cleanup)
        self.clock = FakeClock()
        for target, value in (("_clock", self.clock), ("_sleep", self.clock.sleep)):
            p = patch.object(rlt, target, value)
            p.start()
            self.addCleanup(p.stop)
        rlt._limiters.clear()
        rlt._token_limiters.clear()
        self.addCleanup(rlt._limiters.clear)
        self.addCleanup(rlt._token_limiters.clear)

    def write_providers(self, providers):
        Path(self.home, "config.yaml").write_text(yaml.safe_dump({"providers": providers}))

    def call(self, agent, request=None):
        return rlt.throttle_before_request(agent, agent.provider, agent.model, request)


class NoConfigTest(ThrottleTestCase):
    def test_no_keys_no_wait(self):
        self.write_providers({"acme": {"base_url": "https://acme.example/v1"}})
        agent = FakeAgent()
        for _ in range(5):
            self.assertEqual(self.call(agent, {"messages": [{"content": "x" * 4000}]}), 0.0)
        self.assertEqual(self.clock.sleeps, [])
        self.assertIsNone(rlt.status_snapshot("acme", "m1", agent=FakeAgent()))

    def test_zero_or_negative_is_off(self):
        self.write_providers({"acme": {"requests_per_minute": 0, "input_tokens_per_minute": -5,
                                       "models": {"m1": {"requests_per_minute": -1}}}})
        agent = FakeAgent()
        rlt.record_input_tokens("acme", "m1", 10_000, agent=agent)
        for _ in range(3):
            self.assertEqual(self.call(agent), 0.0)
        self.assertEqual(self.clock.sleeps, [])


class RequestsPerMinuteTest(ThrottleTestCase):
    def test_second_call_waits_about_one_second(self):
        self.write_providers({"acme": {"requests_per_minute": 60}})
        agent = FakeAgent()
        self.assertEqual(self.call(agent), 0.0)
        self.assertAlmostEqual(self.call(agent), 1.0, places=6)
        self.assertEqual(len(self.clock.sleeps), 1)
        self.assertAlmostEqual(self.clock.sleeps[0], 1.0, places=6)
        status = rlt.status_snapshot("acme", "m1", agent=agent)
        self.assertEqual((status.rpm_used, status.rpm_limit, status.tpm_limit), (2, 60.0, None))

    def test_spacing_elapsed_means_no_wait(self):
        self.write_providers({"acme": {"requests_per_minute": 60}})
        agent = FakeAgent()
        self.call(agent)
        self.clock.now += 1.5
        self.assertEqual(self.call(agent), 0.0)

    def test_per_model_value_wins(self):
        self.write_providers({"acme": {"requests_per_minute": 60,
                                       "models": {"slow": {"requests_per_minute": 30}}}})
        slow, other = FakeAgent(model="slow"), FakeAgent(model="other")
        self.call(slow)
        self.assertAlmostEqual(self.call(slow), 2.0, places=6)
        self.call(other)
        self.assertAlmostEqual(self.call(other), 1.0, places=6)

    def test_config_read_once_per_agent(self):
        self.write_providers({"acme": {"requests_per_minute": 6000}})
        real = hconfig.load_config_readonly
        with patch.object(hconfig, "load_config_readonly", side_effect=real) as spy:
            agent = FakeAgent()
            for _ in range(10):
                self.call(agent)
            first = spy.call_count
            for _ in range(10):
                self.call(agent)
        self.assertGreater(first, 0)
        self.assertLessEqual(first, 2)
        self.assertEqual(spy.call_count, first)


class InputTokensPerMinuteTest(ThrottleTestCase):
    def test_recorded_usage_delays_next_request(self):
        self.write_providers({"acme": {"input_tokens_per_minute": 600}})
        agent = FakeAgent()
        self.assertEqual(self.call(agent), 0.0)
        rlt.record_input_tokens("acme", "m1", 1200, agent=agent)
        # 600 tokens over the ceiling drain at 10 tokens/s.
        self.assertAlmostEqual(self.call(agent), 60.0, places=6)
        self.clock.now += 60.0
        self.assertEqual(self.call(agent), 0.0)
        self.assertTrue(any("Pacing" in line for line in agent.printed))

    def test_usage_below_ceiling_no_wait(self):
        self.write_providers({"acme": {"input_tokens_per_minute": 600}})
        agent = FakeAgent()
        rlt.record_input_tokens("acme", "m1", 300, agent=agent)
        self.assertEqual(self.call(agent), 0.0)
        status = rlt.status_snapshot("acme", "m1", agent=agent)
        self.assertEqual((status.tpm_used, status.tpm_limit, status.rpm_limit), (300.0, 600.0, None))


class ProviderKeyValidationTest(unittest.TestCase):
    def test_new_keys_are_known(self):
        entry = {"base_url": "https://acme.example/v1", "requests_per_minute": 60,
                 "input_tokens_per_minute": 100000}
        with self.assertNoLogs("hermes_cli.config", level=logging.WARNING):
            out = _normalize_custom_provider_entry(entry, provider_key="rl-known-keys-test")
        self.assertIsNotNone(out)


if __name__ == "__main__":
    unittest.main()
