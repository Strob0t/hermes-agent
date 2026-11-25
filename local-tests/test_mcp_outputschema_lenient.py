"""Tests of the lenient MCP outputSchema revalidation (tools/mcp_tool_structured_output.py).

The mcp SDK ``ClientSession.call_tool`` revalidates each non-error result against the tool's
declared ``outputSchema`` and raises ``RuntimeError`` when ``structuredContent`` is missing or does
not conform. The patch makes ``MCPServerTask._serve_session`` shadow the session's revalidator
before negotiation: for a tool with a declared schema, a mismatch logs one WARNING per
(server, tool) and the result comes back with its content. The tests show:
 - the installed SDK still has the internals that the hook depends on (``_tool_output_schemas``
   and the ``validate_tool_result`` / ``_validate_tool_result`` method);
 - after ``_serve_session``, a result that does not match its outputSchema (wrong type, or no
   structuredContent) is served with its content instead of a raised error, with 1 WARNING;
 - a result that matches its outputSchema is returned unchanged.
The tests use a real in-process mcp ``Server`` and a real ``ClientSession``.
"""
import asyncio
import inspect
import importlib.metadata
import os
import tempfile
import unittest
from unittest.mock import patch

from mcp import Client, types
from mcp.server.lowlevel import Server

_ENV = patch.dict(os.environ, {"HERMES_HOME": tempfile.mkdtemp(prefix="hermes-home-outputschema-")})


def setUpModule():
    global MCPServerTask
    _ENV.start()
    from tools.mcp_tool import MCPServerTask


def tearDownModule():
    _ENV.stop()


_SCHEMA = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}
_REPLIES = {
    "matching": types.CallToolResult(content=[types.TextContent(type="text", text="count=3")],
                                     structured_content={"count": 3}),
    "wrong_type": types.CallToolResult(content=[types.TextContent(type="text", text="count is null")],
                                       structured_content={"count": None}),
    "text_only": types.CallToolResult(content=[types.TextContent(type="text", text="no_api_units")]),
}


class _StopAfterHook(Exception):
    """Raised in place of the handshake, so that ``_serve_session`` stops after its hook step."""


def _make_server() -> Server:
    async def on_list_tools(_ctx, _params):
        return types.ListToolsResult(tools=[
            types.Tool(name=name, input_schema={"type": "object"}, output_schema=_SCHEMA) for name in _REPLIES])

    async def on_call_tool(_ctx, params):
        return _REPLIES[params.name]

    return Server("outputschema-test", on_list_tools=on_list_tools, on_call_tool=on_call_tool)


async def _call_via_served_session(tool: str):
    """Connect, run ``_serve_session`` up to the handshake, list tools, then call *tool*."""
    async with Client(_make_server()) as client:
        session = client.session
        task = MCPServerTask("srv-" + tool)
        with patch.object(MCPServerTask, "_negotiate_session", side_effect=_StopAfterHook):
            try:
                await task._serve_session(session, 5.0)
            except _StopAfterHook:
                pass
        await session.list_tools()
        return await session.call_tool(tool, {})


class SdkInternalsTest(unittest.TestCase):
    def test_sdk_has_the_internals_the_hook_depends_on(self):
        version = importlib.metadata.version("mcp")

        async def probe():
            async with Client(_make_server()) as client:
                session = client.session
                await session.list_tools()
                names = [n for n in ("validate_tool_result", "_validate_tool_result")
                         if inspect.iscoroutinefunction(getattr(session, n, None))]
                return names, getattr(session, "_tool_output_schemas", None)

        names, schemas = asyncio.run(probe())
        self.assertTrue(names, f"mcp {version}: ClientSession has no async validate_tool_result or "
                               "_validate_tool_result; the lenient outputSchema hook cannot install")
        self.assertIsInstance(schemas, dict, f"mcp {version}: ClientSession has no _tool_output_schemas "
                                             "dict; the lenient outputSchema hook cannot read the declared schema")
        self.assertEqual(schemas.get("matching"), _SCHEMA,
                         f"mcp {version}: _tool_output_schemas does not cache the listed outputSchema")


class LenientOutputSchemaTest(unittest.TestCase):
    def test_non_matching_result_is_served_with_its_content(self):
        for tool, text in (("wrong_type", "count is null"), ("text_only", "no_api_units")):
            with self.subTest(tool=tool), self.assertLogs("tools.mcp_tool", level="WARNING") as logs:
                result = asyncio.run(_call_via_served_session(tool))
                self.assertFalse(result.is_error)
                self.assertEqual([c.text for c in result.content], [text])
                warnings = [r for r in logs.records if r.levelname == "WARNING" and f"'{tool}'" in r.getMessage()]
                self.assertEqual(len(warnings), 1, [r.getMessage() for r in logs.records])

    def test_matching_result_is_unchanged(self):
        result = asyncio.run(_call_via_served_session("matching"))
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content, {"count": 3})
        self.assertEqual([c.text for c in result.content], ["count=3"])


if __name__ == "__main__":
    unittest.main()
