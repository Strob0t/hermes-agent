"""Tests of the MCP progress-timeout port (tools/mcp_tool_loop.py, tools/mcp_tool_handlers.py).

The patch makes the MCP tool timeout an idle timeout: each ``notifications/progress`` from the
server refreshes the deadline, and the total call time stays capped at 4x the configured timeout.
The handler always sends a progress callback (so a ``progressToken``), also with no UI progress sink.
The tests run a real stdio MCP server with ``timeout: 0.4`` in a temp HERMES_HOME and show:
 - a call with no progress times out at about 1x the timeout;
 - a call that reports progress every 0.5x finishes after 2x the timeout; no progress sink is
   bound, and the SDK server sends progress only for a request with a progressToken, so this
   also shows that the token is sent;
 - a call that reports progress all the time is cut at about 4x the timeout.
"""
import json
import os
import sys
import tempfile
import textwrap
import time
import unittest

TIMEOUT = 0.4

SERVER = textwrap.dedent('''\
    import anyio
    from mcp.server.mcpserver import MCPServer, Context
    app = MCPServer("prog")

    async def _ticks(ctx, n, gap):
        for i in range(n):
            await anyio.sleep(gap)
            await ctx.report_progress(i + 1, n, f"tick {i + 1}")

    @app.tool()
    async def silent(ctx: Context) -> str:
        await anyio.sleep(1.2)
        return "silent-done"

    @app.tool()
    async def steady(ctx: Context) -> str:
        await _ticks(ctx, 4, 0.2)
        return "steady-done"

    @app.tool()
    async def endless(ctx: Context) -> str:
        await _ticks(ctx, 40, 0.1)
        return "endless-done"

    app.run()
''')


class McpProgressTimeoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        home = cls._tmp.name
        server = os.path.join(home, "server.py")
        with open(server, "w", encoding="utf-8") as f:
            f.write(SERVER)
        names = ("silent", "steady", "endless")
        cfg = "mcp_servers:\n" + "".join(
            f"  p{n}:\n    command: {json.dumps(sys.executable)}\n    args: [{json.dumps(server)}]\n"
            f"    timeout: {TIMEOUT}\n" for n in names)
        with open(os.path.join(home, "config.yaml"), "w", encoding="utf-8") as f:
            f.write(cfg)
        cls._old_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = home

        from tools import mcp_tool_discovery as disc
        from tools.registry import registry
        cls.registry = registry
        found = disc.discover_mcp_tools()
        for n in names:
            assert f"mcp__p{n}__{n}" in found, found

    @classmethod
    def tearDownClass(cls):
        from tools import mcp_tool_loop as _loop
        from tools.mcp_tool_lifecycle import shutdown_mcp_servers
        shutdown_mcp_servers()
        _loop._stop_mcp_loop()
        if cls._old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = cls._old_home
        cls._tmp.cleanup()

    def call(self, name):
        start = time.monotonic()
        result = self.registry.dispatch(f"mcp__p{name}__{name}", {})
        return result, time.monotonic() - start

    def test_no_progress_times_out_at_1x(self):
        result, elapsed = self.call("silent")
        self.assertNotIn("silent-done", result)
        self.assertIn("timed out", result)
        self.assertLess(elapsed, 2 * TIMEOUT)

    def test_progress_every_half_timeout_finishes_after_2x(self):
        result, elapsed = self.call("steady")
        self.assertIn("steady-done", result)
        self.assertGreaterEqual(elapsed, 2 * TIMEOUT)

    def test_continuous_progress_is_cut_at_4x(self):
        result, elapsed = self.call("endless")
        self.assertNotIn("endless-done", result)
        self.assertIn("timed out", result)
        self.assertGreaterEqual(elapsed, 4 * TIMEOUT - 0.1)
        self.assertLess(elapsed, 4 * TIMEOUT + 0.6)


if __name__ == "__main__":
    unittest.main()
