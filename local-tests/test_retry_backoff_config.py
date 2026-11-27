"""Tests of the configurable retry backoff patch (port of fork commit 866ff8a190f6), run in the image build.

The patch reads ``agent.retry_base_delay``, ``agent.retry_max_delay``,
``agent.rate_limit_retry_base_delay`` and ``agent.rate_limit_retry_max_delay`` (clamped >= 0) in
``agent_init`` and uses them at the retry sites: invalid response (turn_response_check), empty
response (turn_empty_response) and retryable API error / Retry-After cap (turn_recovery). The
trajectory compressor gets ``summarization.retry_max_delay``. The tests show:
 - with no keys, each site calls ``jittered_backoff`` with its old built-in base/max values;
 - with keys set, the sites use the configured values;
 - a negative value is clamped to 0;
 - a provider Retry-After is capped at ``rate_limit_retry_max_delay``;
 - ``summarization.retry_max_delay`` sets the trajectory compressor backoff cap.
Each test uses a temp HERMES_HOME; nothing is written to ``~/.hermes/``.
"""
import logging
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml

from agent.agent_init import _apply_agent_section
from agent.turn_empty_response import _retry_empty
from agent.turn_recovery import compute_error_backoff
from agent.turn_response_check import retry_invalid_response
from agent.turn_retry_state import TurnRetryState
from hermes_cli.config import load_config
from trajectory_compressor import CompressionConfig, TrajectoryCompressor, TrajectoryMetrics


class _Recorder:
    """Stand-in for ``jittered_backoff``: records the delays and returns 0 so no test sleeps."""

    def __init__(self):
        self.calls = []

    def __call__(self, attempt, *, base_delay=5.0, max_delay=120.0, jitter_ratio=0.5):
        self.calls.append((base_delay, max_delay))
        return 0.0


class _TempHomeCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        env = patch.dict(os.environ, {"HERMES_HOME": self._tmp.name})
        env.start()
        self.addCleanup(env.stop)

    def agent_for(self, agent_keys):
        section = {"environment_probe": False, **agent_keys}
        with open(os.path.join(self._tmp.name, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump({"agent": section}, f)
        agent = MagicMock()
        agent.run_budget_seconds = None
        agent.log_prefix, agent.quiet_mode, agent.api_mode = "", True, "chat_completions"
        agent.provider, agent.model = "custom", "test/model"
        agent._fallback_chain, agent._fallback_index = [], 0
        agent._try_activate_fallback.return_value = False
        agent._has_pending_fallback.return_value = False
        agent._client_log_context.return_value = ""
        agent._empty_content_retries = 0
        agent._interrupt_requested = False
        _apply_agent_section(agent, load_config())
        return agent

    def invalid_response_delays(self, agent):
        rec = _Recorder()
        with patch("agent.retry_utils.jittered_backoff", rec):
            retry_invalid_response(
                agent, response=SimpleNamespace(choices=[], error=None),
                error_details=["response.choices is empty"], _retry=TurnRetryState(),
                thinking_spinner=None, messages=[], api_messages=[], api_kwargs=None,
                active_system_prompt=None, conversation_history=None, retry_count=0, max_retries=3,
                compression_attempts=0, api_call_count=1, api_request_id="r", api_start_time=0.0,
                api_duration=0.4, effective_task_id="t", turn_id="turn",
            )
        return rec.calls

    def empty_response_delays(self, agent):
        rec = _Recorder()
        with patch("agent.retry_utils.jittered_backoff", rec):
            _retry_empty(
                agent, SimpleNamespace(usage=None), "stop", True,
                messages=[], conversation_history=None, api_call_count=1,
            )
        return rec.calls

    def error_backoff(self, agent, error=None):
        rec = _Recorder()
        with patch("agent.retry_utils.jittered_backoff", rec):
            wait = compute_error_backoff(
                agent, error or Exception("HTTP 503"), retry_count=1, max_retries=3,
                is_rate_limited=False, is_zai_coding_overload=False,
                base_url="https://example.test/v1", model="test/model",
            )
        return wait, rec.calls


def _retry_after_error(seconds):
    err = Exception("HTTP 503")
    err.response = SimpleNamespace(headers={"Retry-After": str(seconds)})
    return err


class RetryBackoffDefaultsTest(_TempHomeCase):
    def test_no_keys_keep_the_old_delays(self):
        agent = self.agent_for({})
        self.assertEqual(self.invalid_response_delays(agent), [(5.0, 120.0)])
        self.assertEqual(self.empty_response_delays(agent), [(5.0, 60.0)])
        self.assertEqual(self.error_backoff(agent)[1], [(2.0, 60.0)])

    def test_no_keys_keep_the_600s_retry_after_cap(self):
        wait, calls = self.error_backoff(self.agent_for({}), _retry_after_error(900))
        self.assertEqual((wait, calls), (600.0, []))


class RetryBackoffConfiguredTest(_TempHomeCase):
    KEYS = {
        "retry_base_delay": 1.5, "retry_max_delay": 9,
        "rate_limit_retry_base_delay": 0.5, "rate_limit_retry_max_delay": 7,
    }

    def test_set_keys_reach_every_retry_site(self):
        agent = self.agent_for(self.KEYS)
        self.assertEqual(self.invalid_response_delays(agent), [(1.5, 9.0)])
        self.assertEqual(self.empty_response_delays(agent), [(1.5, 9.0)])
        self.assertEqual(self.error_backoff(agent)[1], [(0.5, 7.0)])

    def test_negative_values_are_clamped_to_zero(self):
        agent = self.agent_for({"retry_base_delay": -3, "rate_limit_retry_max_delay": -1})
        self.assertEqual(self.invalid_response_delays(agent), [(0.0, 120.0)])
        self.assertEqual(self.error_backoff(agent)[1], [(2.0, 0.0)])

    def test_retry_after_is_capped_at_rate_limit_retry_max_delay(self):
        agent = self.agent_for(self.KEYS)
        self.assertEqual(self.error_backoff(agent, _retry_after_error(300)), (7.0, []))
        self.assertEqual(self.error_backoff(agent, _retry_after_error(3)), (3.0, []))


class TrajectoryRetryMaxDelayTest(_TempHomeCase):
    def summary_delays(self, config):
        compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
        compressor.config, compressor.logger = config, logging.getLogger("test_retry_backoff_config")
        rec = _Recorder()
        with patch("trajectory_compressor.jittered_backoff", rec), self.assertLogs(compressor.logger, "WARNING"):
            compressor._summary_attempt_failed(TrajectoryMetrics(), 0, RuntimeError("boom"))
        return rec.calls

    def test_default_cap_is_30s(self):
        self.assertEqual(self.summary_delays(CompressionConfig()), [(2, 30.0)])

    def test_yaml_retry_max_delay_sets_the_cap(self):
        path = os.path.join(self._tmp.name, "compress.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"summarization": {"retry_delay": 1, "retry_max_delay": 4}}, f)
        self.assertEqual(self.summary_delays(CompressionConfig.from_yaml(path)), [(1, 4)])


if __name__ == "__main__":
    unittest.main()
