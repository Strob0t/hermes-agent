"""Tests of the MCP tool-listing guard port (tools/mcp_listing_guard.py), run in the image build.

The mcp 2.x SDK validates a ``tools/list`` page as a whole, so one invalid tool parks the whole
server. The patch wraps the session streams before ``ClientSession``: the write proxy records the
method of each outbound request, and the read proxy validates each tool of a ``tools/list`` response
against the SDK's versioned ``Tool`` model and drops only the invalid tools, by name. The server task
records the drops in ``_dropped_tools``. The tests show:
 - a ``tools/list`` response with 1 invalid and 2 valid tools gives the 2 valid tools and 1 dropped name;
 - an all-valid ``tools/list`` response is unchanged and drops nothing;
 - a response to a different method, and requests from the server, pass through unchanged;
 - the server task wiring records the dropped names in ``_dropped_tools``.
"""
import copy
import os
import shutil
import tempfile
import unittest

import anyio
from mcp.shared.message import SessionMessage
from mcp_types.jsonrpc import JSONRPCRequest, JSONRPCResponse

VERSION = "2025-11-25"


def _tool(name, schema=None):
    return {"name": name, "inputSchema": schema if schema is not None else {"type": "object"}}


def _invalid_tool(name):
    return _tool(name, {"type": "string"})  # inputSchema must be an object schema


class GuardTestBase(unittest.TestCase):
    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="hermes-home-")
        self._old_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = self._home

    def tearDown(self):
        if self._old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self._old_home
        shutil.rmtree(self._home, ignore_errors=True)

    def exchange(self, outbound, inbound, guard_factory=None):
        """Send *outbound* through the guarded write stream, then read *inbound* through the guarded read
        stream. Returns (received items, items that reached the transport, drops reported)."""
        from tools.mcp_listing_guard import guard_session_streams

        drops = []

        async def run():
            to_session_w, to_session_r = anyio.create_memory_object_stream(16)
            to_transport_w, to_transport_r = anyio.create_memory_object_stream(16)
            if guard_factory is None:
                read, write = guard_session_streams(
                    to_session_r, to_transport_w, server_name="srv",
                    version_getter=lambda: VERSION, on_drop=drops.append)
            else:
                read, write = guard_factory(to_session_r, to_transport_w)
            sent, got = [], []
            for item in outbound:
                await write.send(item)
                sent.append(await to_transport_r.receive())
            for item in inbound:
                await to_session_w.send(item)
                got.append(await read.receive())
            for s in (to_session_w, to_session_r, to_transport_w, to_transport_r):
                await s.aclose()
            return got, sent

        got, sent = anyio.run(run)
        return got, sent, drops


def _request(rid, method):
    return SessionMessage(message=JSONRPCRequest(jsonrpc="2.0", id=rid, method=method, params={}))


def _response(rid, result):
    return SessionMessage(message=JSONRPCResponse(jsonrpc="2.0", id=rid, result=result))


class ToolListingGuardTest(GuardTestBase):
    def test_one_invalid_tool_is_dropped_and_two_valid_tools_stay(self):
        result = {"tools": [_tool("alpha"), _invalid_tool("broken"), _tool("gamma")]}
        got, _, drops = self.exchange([_request(1, "tools/list")], [_response(1, result)])
        tools = got[0].message.result["tools"]
        self.assertEqual([t["name"] for t in tools], ["alpha", "gamma"])
        self.assertEqual(len(drops), 1)
        self.assertEqual([name for name, _reason in drops[0]], ["broken"])

    def test_all_valid_list_is_unchanged(self):
        result = {"tools": [_tool("alpha"), _tool("beta", {"type": "object", "properties": {"x": {"type": "string"}}})]}
        expected = copy.deepcopy(result)
        got, _, drops = self.exchange([_request(7, "tools/list")], [_response(7, result)])
        self.assertEqual(got[0].message.result, expected)
        self.assertEqual(drops, [])

    def test_other_messages_pass_through_unchanged(self):
        custom = {"tools": [_invalid_tool("broken")]}
        expected = copy.deepcopy(custom)
        outbound = [_request(2, "x-ext/catalog")]
        server_request = _request(99, "tools/list")  # a request from the server, not a response
        inbound = [_response(2, custom), server_request, _response(3, {"tools": [_invalid_tool("b2")]})]
        got, sent, drops = self.exchange(outbound, inbound)
        self.assertIs(sent[0], outbound[0])
        self.assertIs(got[0], inbound[0])
        self.assertEqual(got[0].message.result, expected)
        self.assertIs(got[1], server_request)
        self.assertEqual(len(got[2].message.result["tools"]), 1)  # unknown id: not a tools/list answer
        self.assertEqual(drops, [])

    def test_server_task_records_dropped_tools(self):
        from tools.mcp_tool import MCPServerTask

        task = MCPServerTask("srv")
        self.assertEqual(task._dropped_tools, {})

        class _Session:
            protocol_version = VERSION

        task.session = _Session()
        result = {"tools": [_tool("alpha"), _invalid_tool("broken")]}
        got, _, _ = self.exchange([_request(5, "tools/list")], [_response(5, result)],
                                  guard_factory=task._guard_session_streams)
        self.assertEqual([t["name"] for t in got[0].message.result["tools"]], ["alpha"])
        self.assertEqual(list(task._dropped_tools), ["broken"])


if __name__ == "__main__":
    unittest.main()
