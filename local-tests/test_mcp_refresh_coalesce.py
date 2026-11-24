"""Tests of the MCP tools/list_changed flood guard (tools/mcp_tool_health.py), run in the image build.

The patch keeps at most ONE dynamic tool refresh task per server: ``_schedule_tools_refresh`` only
sets a dirty flag while that task is alive, and ``_refresh_tools_task`` drains the flag with a minimum
interval. After 3 consecutive failures it drops pending demand and sleeps a jittered exponential
cooldown (30s to 600s) before one probe. The tests show:
 - 10 notifications while 1 refresh runs cause at most 1 extra refresh;
 - after failed refreshes, notifications during the cooldown start no new refresh;
 - after the cooldown, a notification refreshes again.
The intervals are patched to small values so the tests run fast.
"""
import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch

import tools.mcp_tool_health as health
from tools.mcp_tool import MCPServerTask

MIN_INTERVAL = 0.01
COOLDOWN = 0.3


class _Base(unittest.TestCase):
    def setUp(self):
        self._home = tempfile.TemporaryDirectory()
        self.addCleanup(self._home.cleanup)
        env = patch.dict(os.environ, {"HERMES_HOME": self._home.name})
        env.start()
        self.addCleanup(env.stop)
        for name, value in (("_REFRESH_MIN_INTERVAL_SECONDS", MIN_INTERVAL),
                            ("_REFRESH_COOLDOWN_BASE_SECONDS", COOLDOWN),
                            ("_REFRESH_COOLDOWN_MAX_SECONDS", COOLDOWN)):
            p = patch.object(health, name, value, create=True)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(health, "_jittered", lambda s: s, create=True)
        p.start()
        self.addCleanup(p.stop)

    @staticmethod
    async def _drain(server):
        for _ in range(200):
            if not server._pending_refresh_tasks:
                return
            await asyncio.gather(*list(server._pending_refresh_tasks), return_exceptions=True)


class CoalesceTest(_Base):
    def test_notifications_during_refresh_cause_at_most_one_extra(self):
        async def scenario():
            server = MCPServerTask("flood")
            release = asyncio.Event()
            calls = []

            async def fake_refresh():
                calls.append(1)
                if len(calls) == 1:
                    await release.wait()

            server._refresh_tools = fake_refresh
            server._schedule_tools_refresh()
            await asyncio.sleep(0)
            for _ in range(10):
                server._schedule_tools_refresh()
                await asyncio.sleep(0)
            self.assertEqual(len(calls), 1)
            release.set()
            await self._drain(server)
            return len(calls)

        self.assertIn(asyncio.run(scenario()), (1, 2))

    def test_idle_server_notification_starts_refresh(self):
        async def scenario():
            server = MCPServerTask("idle")
            calls = []

            async def fake_refresh():
                calls.append(1)

            server._refresh_tools = fake_refresh
            server._schedule_tools_refresh()
            await self._drain(server)
            server._schedule_tools_refresh()
            await self._drain(server)
            return len(calls)

        self.assertEqual(asyncio.run(scenario()), 2)


class FailureCooldownTest(_Base):
    async def _fail_into_cooldown(self, server, calls, state):
        async def fake_refresh():
            calls.append(1)
            if state["fail"]:
                raise RuntimeError("server down")

        server._refresh_tools = fake_refresh
        server._schedule_tools_refresh()
        for _ in range(100):
            if len(calls) >= 3:
                break
            server._schedule_tools_refresh()
            await asyncio.sleep(MIN_INTERVAL / 2)

    def test_no_refresh_during_cooldown(self):
        async def scenario():
            server = MCPServerTask("down")
            calls, state = [], {"fail": True}
            with self.assertLogs("tools.mcp_tool", level="WARNING"):
                await self._fail_into_cooldown(server, calls, state)
                await asyncio.sleep(MIN_INTERVAL * 3)
                after_failures = len(calls)
                for _ in range(10):
                    server._schedule_tools_refresh()
                    await asyncio.sleep(COOLDOWN / 40)
                during_cooldown = len(calls)
                server._refresh_task.cancel()
                await self._drain(server)
            return after_failures, during_cooldown

        after_failures, during_cooldown = asyncio.run(scenario())
        self.assertEqual(after_failures, 3)
        self.assertEqual(during_cooldown, 3)

    def test_notification_after_cooldown_refreshes_again(self):
        async def scenario():
            server = MCPServerTask("revived")
            calls, state = [], {"fail": True}
            with self.assertLogs("tools.mcp_tool", level="WARNING"):
                await self._fail_into_cooldown(server, calls, state)
                state["fail"] = False
                await asyncio.sleep(COOLDOWN * 1.5)
                await self._drain(server)
            before = len(calls)
            server._schedule_tools_refresh()
            await self._drain(server)
            return before, len(calls), server._refresh_failures

        before, after, failures = asyncio.run(scenario())
        self.assertEqual(after, before + 1)
        self.assertEqual(failures, 0)


if __name__ == "__main__":
    unittest.main()
