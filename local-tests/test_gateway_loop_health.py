"""Tests of the gateway event-loop probe health port (source fork f3f783b6986e), run in the image build.

The patch adds ``gateway/loop_health.py``, which keeps the last liveness ``ProbeSnapshot``
(consecutive misses and response time) under a lock. The loop liveness watchdog records each probe
result there, and the gateway health export appends ``hermes.gateway.loop.responsive``,
``hermes.gateway.loop.consecutive_misses`` and ``hermes.gateway.loop.response_seconds``.
A loop liveness error text is classified as ``loop_unresponsive``. The tests show:
 - after 3 recorded probes with known lag values, the export holds the values of the last probe;
 - with no recorded probe, the export has no loop metrics and is otherwise unchanged;
 - the real watchdog records answered probes (0 misses) and counts missed probes;
 - a loop liveness error text classifies as ``loop_unresponsive``.
"""
import asyncio
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

_HOME = tempfile.mkdtemp(prefix="hermes-loop-health-")
os.environ["HERMES_HOME"] = _HOME
_ENV = patch.dict(os.environ, {"HERMES_HOME": _HOME})

from agent.monitoring import gateway_health_export as export  # noqa: E402
from agent.monitoring.gateway_health import classify_gateway_error  # noqa: E402
from gateway import loop_health  # noqa: E402
from gateway.shutdown_watchdog import start_loop_liveness_watchdog  # noqa: E402

LOOP_NAMES = (
    "hermes.gateway.loop.responsive",
    "hermes.gateway.loop.consecutive_misses",
    "hermes.gateway.loop.response_seconds",
)


def exported():
    """Return the (name, value) pairs of the runtime health snapshot."""
    return [(m.name, m.value) for m in export._read_runtime_snapshot({}).metrics]


def wait_for(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


class EnvCase(unittest.TestCase):
    def setUp(self):
        _ENV.start()
        self.addCleanup(_ENV.stop)
        loop_health.reset_probe()
        self.addCleanup(loop_health.reset_probe)


class ExportTest(EnvCase):
    def test_three_probes_export_the_last_values(self):
        for misses, lag in ((0, 0.125), (1, 10.5), (2, 10.25)):
            loop_health.record_probe(misses=misses, response_seconds=lag)
        values = dict(exported())
        self.assertEqual(values["hermes.gateway.loop.responsive"], 0)
        self.assertEqual(values["hermes.gateway.loop.consecutive_misses"], 2)
        self.assertEqual(values["hermes.gateway.loop.response_seconds"], 10.25)
        for name in LOOP_NAMES:
            self.assertIn(name, export._OBSERVABLE_METRIC_NAMES)

    def test_recovered_probe_exports_responsive(self):
        for misses, lag in ((1, 10.5), (2, 10.25), (0, 0.0625)):
            loop_health.record_probe(misses=misses, response_seconds=lag)
        values = dict(exported())
        self.assertEqual(values["hermes.gateway.loop.responsive"], 1)
        self.assertEqual(values["hermes.gateway.loop.consecutive_misses"], 0)
        self.assertEqual(values["hermes.gateway.loop.response_seconds"], 0.0625)

    def test_no_probe_leaves_export_unchanged(self):
        before = [name for name, _ in exported()]
        self.assertFalse(set(LOOP_NAMES) & set(before))
        loop_health.record_probe(misses=0, response_seconds=0.5)
        with_probe = [name for name, _ in exported()]
        loop_health.reset_probe()
        after = [name for name, _ in exported()]
        self.assertEqual(after, before)
        self.assertEqual([n for n in with_probe if n not in LOOP_NAMES], before)


class _DeafLoop:
    """Loop double that accepts the probe callback but never runs it (a frozen loop)."""

    def call_soon_threadsafe(self, _cb):
        return None


class WatchdogRecordTest(EnvCase):
    def test_answered_probes_record_zero_misses(self):
        loop = asyncio.new_event_loop()
        runner = threading.Thread(target=loop.run_forever, daemon=True)
        runner.start()
        self.addCleanup(loop.close)
        self.addCleanup(runner.join, 5)
        self.addCleanup(loop.call_soon_threadsafe, loop.stop)
        handle = start_loop_liveness_watchdog(loop, probe_interval=0.02, probe_timeout=1.0, max_strikes=1000)
        self.addCleanup(handle.join, 5)
        self.addCleanup(handle.stop)
        self.assertTrue(wait_for(lambda: loop_health.probe_snapshot() is not None))
        snap = loop_health.probe_snapshot()
        self.assertEqual(snap.misses, 0)
        self.assertGreaterEqual(snap.response_seconds, 0.0)
        self.assertLess(snap.response_seconds, 1.0)

    def test_missed_probes_count_up(self):
        with patch("gateway.shutdown_watchdog._hard_exit") as hard_exit:
            handle = start_loop_liveness_watchdog(
                _DeafLoop(), probe_interval=0.01, probe_timeout=0.03, max_strikes=1000)
            try:
                ok = wait_for(lambda: (loop_health.probe_snapshot() or loop_health.ProbeSnapshot(0, 0)).misses >= 2)
            finally:
                handle.stop()
                handle.join(5)
        self.assertTrue(ok)
        snap = loop_health.probe_snapshot()
        self.assertGreaterEqual(snap.misses, 2)
        self.assertGreaterEqual(snap.response_seconds, 0.03)
        hard_exit.assert_not_called()


class ClassifyTest(unittest.TestCase):
    def test_liveness_text_is_loop_unresponsive(self):
        text = "Gateway event loop missed 3 consecutive liveness probes"
        self.assertEqual(classify_gateway_error(text), "loop_unresponsive")
        self.assertEqual(classify_gateway_error("request timed out"), "timeout")


if __name__ == "__main__":
    unittest.main()
