"""Tests of the tool SystemExit port (fork d31f39ad4983), run in the image build.

The patch makes the tool boundaries catch ``(Exception, SystemExit)`` instead of only
``Exception``: ``ToolRegistry.dispatch``, ``MemoryManager.handle_tool_call`` and the sequential and
concurrent executor paths in ``agent/tool_executor.py``. A handler that calls ``sys.exit()`` then
gives a tool error and the agent keeps running. The tests show:
 - a registry handler that calls ``sys.exit(3)`` returns a tool error with the code; the caller
   keeps running;
 - a memory-provider tool that calls ``sys.exit(3)`` returns a tool error with the code;
 - ``_run_sequential_call`` turns a ``sys.exit(3)`` from the execute callable into the dispatch's
   error result; a dispatch with no error policy (inline/delegate) still re-raises it;
 - ``KeyboardInterrupt`` still propagates out of the registry dispatch and the memory manager;
 - a normal handler result is unchanged.
"""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import agent.tool_executor as te
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from tools.registry import ToolRegistry

SCHEMA = {"name": "exit_tool", "description": "test", "parameters": {"type": "object", "properties": {}}}


class TempHome(unittest.TestCase):
    """Each test runs with its own HERMES_HOME, never ``~/.hermes``."""

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="hermes-sysexit-")
        env = patch.dict(os.environ, {"HERMES_HOME": self.home})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(shutil.rmtree, self.home, True)


def _registry(handler):
    reg = ToolRegistry()
    reg.register("exit_tool", "local_test", SCHEMA, handler)
    return reg


def _exit3(args, **kwargs):
    sys.exit(3)


def _interrupt(args, **kwargs):
    raise KeyboardInterrupt


class RegistryDispatchTests(TempHome):
    def test_sys_exit_becomes_tool_error_with_code(self):
        reg = _registry(_exit3)
        result = reg.dispatch("exit_tool", {})
        payload = json.loads(result)
        self.assertIn("error", payload)
        self.assertIn("SystemExit", payload["error"])
        self.assertIn("3", payload["error"])
        # The caller keeps running: a second call on the same registry works too.
        self.assertIn("error", json.loads(reg.dispatch("exit_tool", {})))

    def test_keyboard_interrupt_still_propagates(self):
        with self.assertRaises(KeyboardInterrupt):
            _registry(_interrupt).dispatch("exit_tool", {})

    def test_normal_result_unchanged(self):
        result = _registry(lambda args, **kw: json.dumps({"ok": 1})).dispatch("exit_tool", {})
        self.assertEqual(json.loads(result), {"ok": 1})


class _Provider(MemoryProvider):
    def __init__(self, behavior):
        self.behavior = behavior

    @property
    def name(self):
        return "localtest"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return [dict(SCHEMA, name="mem_exit_tool")]

    def handle_tool_call(self, tool_name, args, **kwargs):
        return self.behavior()


class MemoryManagerTests(TempHome):
    def _manager(self, behavior):
        mgr = MemoryManager()
        mgr.add_provider(_Provider(behavior))
        self.assertTrue(mgr.has_tool("mem_exit_tool"))
        return mgr

    def test_sys_exit_becomes_tool_error_with_code(self):
        result = self._manager(lambda: sys.exit(3)).handle_tool_call("mem_exit_tool", {})
        payload = json.loads(result)
        self.assertIn("error", payload)
        self.assertIn("3", payload["error"])

    def test_keyboard_interrupt_still_propagates(self):
        def boom():
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self._manager(boom).handle_tool_call("mem_exit_tool", {})

    def test_normal_result_unchanged(self):
        result = self._manager(lambda: json.dumps({"ok": 2})).handle_tool_call("mem_exit_tool", {})
        self.assertEqual(json.loads(result), {"ok": 2})


def _run_direct(agent, *, execute, **kwargs):
    """Stand-in for the sequential middleware runner: calls the tool directly, so the test drives
    the real ``_run_sequential_call`` error policy without the deadline worker thread."""
    args = kwargs["function_args"]
    return te._ManagedToolResult(result=execute(args), args=args, middleware_trace=[], blocked=False, dispatched=True)


class SequentialCallTests(TempHome):
    def _run(self, dispatch):
        agent = SimpleNamespace(_delegate_spinner=None, quiet_mode=False)
        ref = te._ToolCallRef("exit_tool", {}, "task", "call-1", [])
        with patch.object(te, "_run_sequential_tool_execution_middleware", _run_direct):
            managed, _ = te._run_sequential_call(
                agent, dispatch, ref, scope_block=None, messages=[], remaining_calls=[],
                display_index=1, tool_start_time=time.time())
        return managed

    def _dispatch(self, execute, error_result=lambda e: f"Error executing tool 'exit_tool': {e!r}"):
        return te._SequentialDispatch(
            execute=execute, error_result=error_result, error_log="raised for %s: %s",
            finish_spinner=False)

    def test_sys_exit_becomes_error_result_and_agent_keeps_running(self):
        managed = self._run(self._dispatch(lambda args: sys.exit(3)))
        self.assertIn("SystemExit(3)", managed.result)
        self.assertFalse(managed.dispatched)
        # A next call in the same turn still runs.
        self.assertEqual(self._run(self._dispatch(lambda args: "ok")).result, "ok")

    def test_registry_dispatch_through_executor_returns_tool_error(self):
        reg = _registry(_exit3)
        managed = self._run(self._dispatch(lambda args: reg.dispatch("exit_tool", args)))
        self.assertIn("SystemExit", json.loads(managed.result)["error"])

    def test_no_error_policy_still_reraises(self):
        with self.assertRaises(SystemExit):
            self._run(self._dispatch(lambda args: sys.exit(3), error_result=None))

    def test_keyboard_interrupt_still_propagates(self):
        def boom(args):
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self._run(self._dispatch(boom))

    def test_normal_result_unchanged(self):
        self.assertEqual(self._run(self._dispatch(lambda args: '{"ok": 3}')).result, '{"ok": 3}')


if __name__ == "__main__":
    unittest.main()
