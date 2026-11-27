"""Tests of the memory-provider init-failure port (agent/memory_manager.py), run in the image build.

The patch runs each provider's ``initialize`` in its own try block inside ``MemoryManager.initialize_all``.
A provider that raises is dropped from the manager and shut down; only the error type (never the
exception message) is recorded in a lock-protected dict, read through the new property
``provider_initialization_failures``. ``initialize_for_agent`` uses that dict to warn that the provider
"failed to initialize" instead of the shared-instance warning. The tests show:
 - provider A raises in ``initialize`` -> A is removed and shut down, its tools are gone, provider B
   stays active and initialized;
 - the failure dict holds A with the error type only, no message text (also for a failing shutdown);
 - with no failure the dict is empty and the provider list is unchanged;
 - ``initialize_for_agent`` reports "failed to initialize" without the exception text.
"""
import json
import os
import shutil
import tempfile
import unittest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider

SECRET = "secret initialization detail"


class FakeProvider(MemoryProvider):
    def __init__(self, name, tools=None, init_error=None, shutdown_error=None):
        self._name = name
        self._tools = tools or []
        self._init_error = init_error
        self._shutdown_error = shutdown_error
        self.initialized = False
        self.shutdown_calls = 0
        self.turn_starts = []

    @property
    def name(self):
        return self._name

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        if self._init_error is not None:
            raise self._init_error
        self.initialized = True

    def get_tool_schemas(self):
        return self._tools

    def handle_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({"handled": tool_name})

    def on_turn_start(self, turn_number, message, **kwargs):
        self.turn_starts.append(turn_number)

    def shutdown(self):
        self.shutdown_calls += 1
        if self._shutdown_error is not None:
            raise self._shutdown_error


class InitFailureTest(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="hermes-home-")
        self._old_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = self.home

    def tearDown(self):
        if self._old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self._old_home
        shutil.rmtree(self.home, ignore_errors=True)

    def _manager(self, a, b):
        mgr = MemoryManager()
        mgr.add_provider(b)
        mgr.add_provider(a)
        return mgr

    def test_failing_provider_is_dropped_and_shut_down(self):
        a = FakeProvider("ext_a", tools=[{"name": "a_tool", "description": "A", "parameters": {}}],
                         init_error=RuntimeError(SECRET))
        b = FakeProvider("builtin", tools=[{"name": "b_tool", "description": "B", "parameters": {}}])
        mgr = self._manager(a, b)

        with self.assertLogs("agent.memory_manager", level="ERROR") as logs:
            mgr.initialize_all(session_id="s", hermes_home=self.home)

        self.assertEqual([p.name for p in mgr.providers], ["builtin"])
        self.assertEqual(a.shutdown_calls, 1)
        self.assertTrue(b.initialized)
        self.assertFalse(mgr.has_tool("a_tool"))
        self.assertEqual(json.loads(mgr.handle_tool_call("b_tool", {}))["handled"], "b_tool")
        mgr.on_turn_start(1, "hi")
        self.assertEqual(b.turn_starts, [1])
        self.assertEqual(a.turn_starts, [])
        self.assertNotIn(SECRET, "\n".join(logs.output))

    def test_failure_dict_holds_error_type_only(self):
        a = FakeProvider("ext_a", init_error=RuntimeError(SECRET), shutdown_error=ValueError("secret cleanup"))
        mgr = self._manager(a, FakeProvider("builtin"))

        with self.assertLogs("agent.memory_manager", level="ERROR"):
            mgr.initialize_all(session_id="s", hermes_home=self.home)

        failures = mgr.provider_initialization_failures
        self.assertEqual(failures, {"ext_a": {"status": "failed", "error_type": "RuntimeError",
                                              "cleanup_error_type": "ValueError"}})
        self.assertNotIn(SECRET, repr(failures))
        self.assertNotIn("secret cleanup", repr(failures))
        failures["ext_a"]["error_type"] = "changed"
        self.assertEqual(mgr.provider_initialization_failures["ext_a"]["error_type"], "RuntimeError")

    def test_no_failure_keeps_providers_and_empty_dict(self):
        a, b = FakeProvider("ext_a"), FakeProvider("builtin")
        mgr = self._manager(a, b)
        before = mgr.providers

        mgr.initialize_all(session_id="s", hermes_home=self.home)

        self.assertEqual(mgr.provider_initialization_failures, {})
        self.assertEqual(len(mgr.providers), len(before))
        self.assertTrue(all(x is y for x, y in zip(mgr.providers, before)))
        self.assertTrue(a.initialized and b.initialized)
        self.assertEqual(a.shutdown_calls, 0)

    def test_initialize_for_agent_warns_failed_to_initialize(self):
        mgr = self._manager(FakeProvider("ext_a", init_error=RuntimeError(SECRET)), FakeProvider("builtin"))
        warnings = []

        with self.assertLogs("agent.memory_manager", level="ERROR"):
            served = mgr.initialize_for_agent(say=warnings.append, session_id="s", hermes_home=self.home)

        self.assertFalse(served)
        self.assertEqual(len(warnings), 1)
        self.assertIn("failed to initialize", warnings[0])
        self.assertNotIn("another profile already uses", warnings[0])
        self.assertNotIn(SECRET, warnings[0])


if __name__ == "__main__":
    unittest.main()
