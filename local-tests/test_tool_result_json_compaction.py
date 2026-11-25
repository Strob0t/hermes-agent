"""Tests of the lossless JSON compaction of tool results (tools/tool_result_compaction.py).

The patch removes whitespace outside JSON string literals from a valid JSON tool result in
``agent.tool_executor._commit_tool_result`` (after spillover). The policy is
``tool_output.json_compaction`` with mode off / observe / compact (default compact); it is read
from the config once per process and cached. The tests show:
 - compact mode: pretty JSON has no whitespace between tokens and ``json.loads`` is equal;
 - observe mode: the content is unchanged and the savings are logged;
 - off mode: the content is unchanged;
 - non-JSON text is unchanged;
 - 10 calls read the config once.
"""
import json
import os
import re
import tempfile
import unittest
from unittest.mock import patch

import hermes_cli.config as hconfig
import tools.tool_result_compaction as trc
from hermes_cli.config_defaults import DEFAULT_CONFIG

PRETTY = json.dumps(
    {"records": [{"id": i, "title": "Meeting  at 10", "note": "a\n  b", "active": True}
                 for i in range(40)]},
    indent=2,
)


class _HermesHome(unittest.TestCase):
    """Run each test with a temp HERMES_HOME and a fresh policy cache."""

    mode = None

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        env = {"HERMES_HOME": self._tmp.name}
        self._env = patch.dict(os.environ, env)
        self._env.start()
        if self.mode is not None:
            with open(os.path.join(self._tmp.name, "config.yaml"), "w", encoding="utf-8") as fh:
                # JSON is valid YAML.
                json.dump({"tool_output": {"json_compaction": {"mode": self.mode}}}, fh)
        trc.reset_policy_cache()

    def tearDown(self):
        trc.reset_policy_cache()
        self._env.stop()
        self._tmp.cleanup()


class CompactModeTest(_HermesHome):
    mode = "compact"

    def test_pretty_json_loses_whitespace_between_tokens_and_value_is_equal(self):
        out = trc.compact_tool_result(PRETTY, "web_search")
        self.assertLess(len(out), len(PRETTY))
        self.assertEqual(json.loads(out), json.loads(PRETTY))
        outside_strings = re.sub(r'"(?:[^"\\]|\\.)*"', '""', out)
        self.assertIsNone(re.search(r"[ \t\r\n]", outside_strings))

    def test_string_literals_are_kept_byte_for_byte(self):
        out = trc.compact_tool_result(PRETTY, "web_search")
        self.assertIn('"Meeting  at 10"', out)
        self.assertIn('"a\\n  b"', out)


class DefaultModeTest(_HermesHome):
    def test_default_mode_is_compact(self):
        self.assertEqual(DEFAULT_CONFIG["tool_output"]["json_compaction"]["mode"], "compact")
        out = trc.compact_tool_result(PRETTY, "web_search")
        self.assertLess(len(out), len(PRETTY))
        self.assertEqual(json.loads(out), json.loads(PRETTY))


class ObserveModeTest(_HermesHome):
    mode = "observe"

    def test_content_unchanged_and_savings_logged(self):
        with self.assertLogs("tools.tool_result_compaction", level="INFO") as logs:
            out = trc.compact_tool_result(PRETTY, "web_search")
        self.assertEqual(out, PRETTY)
        line = "\n".join(logs.output)
        self.assertIn("mode=observe", line)
        self.assertIn(f"original_chars={len(PRETTY)}", line)
        self.assertRegex(line, r"compact_chars=\d+")


class OffModeTest(_HermesHome):
    mode = "off"

    def test_content_unchanged(self):
        self.assertEqual(trc.compact_tool_result(PRETTY, "web_search"), PRETTY)


class NonJsonTest(_HermesHome):
    mode = "compact"

    def test_non_json_text_unchanged(self):
        texts = [
            "line one\n    indented line two\n" * 100,
            "{ not json at all,\n   just braces }" + " " * 2000,
            PRETTY[:-5],
        ]
        for text in texts:
            self.assertEqual(trc.compact_tool_result(text, "terminal"), text)


class ConfigCacheTest(_HermesHome):
    mode = "compact"

    def test_config_read_once_for_ten_calls(self):
        real = hconfig.load_config_readonly
        calls = []

        def counting():
            calls.append(1)
            return real()

        with patch.object(hconfig, "load_config_readonly", counting):
            outs = [trc.compact_tool_result(PRETTY, "web_search") for _ in range(10)]
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(len(o) < len(PRETTY) for o in outs))


if __name__ == "__main__":
    unittest.main()
