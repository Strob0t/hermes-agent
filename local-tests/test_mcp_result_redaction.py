"""Tests of the opt-in secret redaction of MCP results (tools/mcp_tool_handlers.py).

The patch sends each MCP handler result through ``_mcp_result_json``. When
``security.redact_mcp_results`` is true (default false), every string value of the payload is
passed through ``redact_sensitive_text(file_read=True)`` before ``json.dumps``: tool-result text,
``structuredContent``, ``_meta``, ``read_resource`` results and ``get_prompt`` message content.
The tests show:
 - default config: a result with a fake API key is byte-identical to plain ``json.dumps``;
 - enabled: the key is masked in text, structuredContent, _meta, a resource read and a prompt message;
 - enabled: a normal value stays unchanged and the output stays valid JSON.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from mcp import types as mt

import tools.mcp_tool_handlers as h
from hermes_cli.config_defaults import DEFAULT_CONFIG

FAKE_KEY = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
NORMAL = "Meeting at 10, room \"B\"\nline two"


def _tool_result():
    return mt.CallToolResult(
        content=[mt.TextContent(type="text", text=f"token is {FAKE_KEY}")],
        structuredContent={"auth": {"token": FAKE_KEY}, "note": NORMAL, "count": 3},
        _meta={"debug": f"used {FAKE_KEY}", "note": NORMAL},
    )


def _resource_result():
    return mt.ReadResourceResult(contents=[
        mt.TextResourceContents(uri="file:///x.txt", mimeType="text/plain", text=f"key={FAKE_KEY}\n{NORMAL}")])


def _prompt_result():
    return mt.GetPromptResult(description="p", messages=[
        mt.PromptMessage(role="user", content=mt.TextContent(type="text", text=f"use {FAKE_KEY} now"))])


class _HermesHome(unittest.TestCase):
    """Run each test with a temp HERMES_HOME; ``enabled`` writes the config key."""

    enabled = None

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = patch.dict(os.environ, {"HERMES_HOME": self._tmp.name})
        self._env.start()
        if self.enabled is not None:
            with open(os.path.join(self._tmp.name, "config.yaml"), "w", encoding="utf-8") as fh:
                json.dump({"security": {"redact_mcp_results": self.enabled}}, fh)

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def tool_out(self):
        return h._render_call_tool_result(_tool_result(), "srv")

    def resource_out(self):
        return h._mcp_result_json(h._render_read_resource(_resource_result(), "srv"))

    def prompt_out(self):
        return h._mcp_result_json(h._render_get_prompt(_prompt_result(), "srv"))


class DefaultConfigTest(_HermesHome):
    def test_default_key_is_false(self):
        self.assertIs(DEFAULT_CONFIG["security"]["redact_mcp_results"], False)

    def test_tool_result_is_unchanged(self):
        out = self.tool_out()
        self.assertEqual(out.count(FAKE_KEY), 3)
        expected = {"result": f"token is {FAKE_KEY}",
                    "structuredContent": {"auth": {"token": FAKE_KEY}, "note": NORMAL, "count": 3},
                    "_meta": {"debug": f"used {FAKE_KEY}", "note": NORMAL}}
        self.assertEqual(out, json.dumps(expected, ensure_ascii=False))

    def test_resource_and_prompt_are_unchanged(self):
        self.assertEqual(self.resource_out(),
                         json.dumps({"result": f"key={FAKE_KEY}\n{NORMAL}"}, ensure_ascii=False))
        self.assertEqual(json.loads(self.prompt_out())["messages"][0]["content"], f"use {FAKE_KEY} now")


class ExplicitFalseTest(DefaultConfigTest):
    enabled = False


class EnabledTest(_HermesHome):
    enabled = True

    def test_tool_result_masks_text_structured_and_meta(self):
        out = self.tool_out()
        self.assertNotIn(FAKE_KEY, out)
        data = json.loads(out)
        self.assertTrue(data["result"].startswith("token is "))
        self.assertNotIn(FAKE_KEY, data["result"])
        self.assertNotIn(FAKE_KEY, data["structuredContent"]["auth"]["token"])
        self.assertNotIn(FAKE_KEY, data["_meta"]["debug"])
        self.assertIn("redacted", data["structuredContent"]["auth"]["token"])

    def test_normal_values_and_types_stay_unchanged(self):
        data = json.loads(self.tool_out())
        self.assertEqual(data["structuredContent"]["note"], NORMAL)
        self.assertEqual(data["structuredContent"]["count"], 3)
        self.assertEqual(data["_meta"]["note"], NORMAL)
        self.assertEqual(list(data), ["result", "structuredContent", "_meta"])

    def test_resource_read_is_masked(self):
        out = self.resource_out()
        self.assertNotIn(FAKE_KEY, out)
        self.assertTrue(json.loads(out)["result"].endswith(NORMAL))

    def test_prompt_message_is_masked(self):
        data = json.loads(self.prompt_out())
        self.assertNotIn(FAKE_KEY, data["messages"][0]["content"])
        self.assertEqual(data["messages"][0]["role"], "user")
        self.assertEqual(data["description"], "p")

    def test_text_only_result_is_masked_and_valid_json(self):
        r = mt.CallToolResult(content=[mt.TextContent(type="text", text=f'{{"k": "{FAKE_KEY}"}}')])
        data = json.loads(h._render_call_tool_result(r, "srv"))
        self.assertNotIn(FAKE_KEY, data["result"])

    def test_non_string_payload_passes_through(self):
        payload = {"a": [1, 2.5, None, True], "b": "plain"}
        self.assertEqual(h._mcp_result_json(payload), json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
