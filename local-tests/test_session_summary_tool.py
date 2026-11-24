"""Tests of the opt-in session_summary tool (tools/session_summary_tool.py), run in the image build.

The patch adds a ``session_summary`` tool with ``write``, ``append`` (``## HH:MM`` entry) and ``read``
on ``<HERMES_HOME>/sessions/<id>/running_summary.md``. It is in its own ``session_summary`` toolset,
not in the core tools, the platform bundles or ``session_search``. The tests show:
 - the default toolsets and the default CLI config do not give the tool to an agent;
 - with the toolset enabled, write/append/read work through the registry in a temp HERMES_HOME;
 - the current session id is used, and a session_id such as ``../x`` is rejected with no file outside;
 - an append that makes the file larger than MAX_SUMMARY_BYTES is refused and the file stays the same.
Each test uses its own temp HERMES_HOME; nothing is written to ``~/.hermes``.
"""
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class SessionSummaryToolTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name) / "hermes"
        self.home.mkdir()
        env = patch.dict(os.environ, {"HERMES_HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("HERMES_SESSION_ID", None)
        self.addCleanup(self._tmp.cleanup)

    def _dispatch(self, args, session_id="sess-1"):
        import tools.session_summary_tool  # noqa: F401  (registers the tool)
        from tools.registry import registry
        return json.loads(registry.dispatch("session_summary", args, session_id=session_id))

    def test_default_toolsets_do_not_contain_session_summary(self):
        from toolsets import resolve_toolset
        from hermes_cli.tools_config import _get_platform_tools
        for name in ("hermes-cli", "hermes-api-server", "hermes-telegram", "session_search", "coding"):
            self.assertNotIn("session_summary", resolve_toolset(name), name)
        self.assertNotIn("session_summary", _get_platform_tools({}, "cli"))
        self.assertEqual(resolve_toolset("session_summary"), ["session_summary"])
        enabled = _get_platform_tools({"platform_toolsets": {"cli": ["hermes-cli", "session_summary"]}}, "cli")
        self.assertIn("session_summary", enabled)

    def test_enabled_toolset_write_append_read(self):
        import model_tools
        names = [t["function"]["name"] for t in model_tools.get_tool_definitions(
            enabled_toolsets=["session_summary"], quiet_mode=True, skip_tool_search_assembly=True)]
        self.assertIn("session_summary", names)

        res = self._dispatch({"action": "read"})
        self.assertEqual((res["success"], res["content"]), (True, ""))
        res = self._dispatch({"action": "write", "content": "# Goal\nport 106"})
        self.assertTrue(res["success"], res)
        path = self.home / "sessions" / "sess-1" / "running_summary.md"
        self.assertEqual(Path(res["path"]), path)
        res = self._dispatch({"action": "append", "content": "fixed the bug"})
        self.assertTrue(res["success"], res)
        text = self._dispatch({"action": "read"})["content"]
        self.assertEqual(text, path.read_text(encoding="utf-8"))
        self.assertRegex(text, re.compile(r"\A# Goal\nport 106\n\n## \d\d:\d\d\nfixed the bug\n\Z"))

    def test_current_session_id_wins_over_argument(self):
        res = self._dispatch({"action": "write", "content": "x", "session_id": "other"}, session_id="cur")
        self.assertTrue(res["success"], res)
        self.assertTrue((self.home / "sessions" / "cur" / "running_summary.md").is_file())
        self.assertFalse((self.home / "sessions" / "other").exists())

    def test_traversal_session_id_is_rejected(self):
        for bad in ("../x", "..", "a/b", "a\\b"):
            for action in ("write", "append", "read"):
                res = self._dispatch({"action": action, "content": "pwn", "session_id": bad}, session_id=None)
                self.assertFalse(res["success"], (bad, action, res))
        self.assertFalse((self.home / "x").exists())
        self.assertFalse((self.home / "sessions").exists())

    def test_append_above_size_limit_is_refused(self):
        from tools.session_summary_tool import MAX_SUMMARY_BYTES
        self.assertLessEqual(MAX_SUMMARY_BYTES, 1024 * 1024)
        self.assertTrue(self._dispatch({"action": "write", "content": "a" * (MAX_SUMMARY_BYTES - 10)})["success"])
        before = self._dispatch({"action": "read"})["content"]
        res = self._dispatch({"action": "append", "content": "b" * 100})
        self.assertFalse(res["success"], res)
        self.assertIn("limit", res["error"])
        self.assertEqual(self._dispatch({"action": "read"})["content"], before)
        self.assertFalse(self._dispatch({"action": "write", "content": "c" * (MAX_SUMMARY_BYTES + 1)})["success"])


if __name__ == "__main__":
    unittest.main()
