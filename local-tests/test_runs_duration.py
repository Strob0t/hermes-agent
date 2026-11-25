"""Tests of the local runs-duration patch (upstream-pr-149737.diff), run in the image build.

Before the patch the ``/v1/runs`` ``tool.completed`` SSE event carried ``duration`` 0 and an empty
preview, because ``_tool_complete`` published neither the elapsed time nor the result. The gateway
part of upstream PR #149737 records the start time at ``tool.started`` and passes ``result`` and
``duration`` to the completion callback. These tests pin the wire-format seam the diff feeds: the
fixed-field builder for ``tool.completed`` and the bounded, secret-redacted preview.
"""
import unittest

from gateway.platforms.api_server_runs import (
    _FIXED_EVENT_FIELDS, _TOOL_COMPLETED_PREVIEW_MAX_CHARS, _tool_completed_preview,
)


def identity_redact(text, force=False):
    return text


class ToolCompletedFieldsTest(unittest.TestCase):
    def build(self, **kwargs):
        return _FIXED_EVENT_FIELDS["tool.completed"]("terminal", None, kwargs)

    def test_duration_is_rounded_and_carried(self):
        fields = self.build(duration=1.23456, is_error=False, tool_call_id="c1")
        self.assertEqual(fields["duration"], 1.235)
        self.assertEqual(fields["tool"], "terminal")
        self.assertFalse(fields["error"])
        self.assertEqual(fields["tool_call_id"], "c1")

    def test_missing_duration_defaults_to_zero(self):
        fields = self.build(is_error=False)
        self.assertEqual(fields["duration"], 0)

    def test_error_flag_is_passed_through(self):
        fields = self.build(duration=0.5, is_error=True, tool_call_id="c2")
        self.assertTrue(fields["error"])

    def test_no_call_id_omits_the_field(self):
        fields = self.build(duration=0.5, is_error=False)
        self.assertNotIn("tool_call_id", fields)


class ToolCompletedPreviewTest(unittest.TestCase):
    def test_none_result_is_empty(self):
        self.assertEqual(_tool_completed_preview(None, identity_redact), "")

    def test_str_result_is_kept(self):
        self.assertEqual(_tool_completed_preview("done", identity_redact), "done")

    def test_structured_result_is_json(self):
        out = _tool_completed_preview({"ok": True}, identity_redact)
        self.assertIn("\"ok\"", out)

    def test_long_result_is_truncated_with_ellipsis(self):
        out = _tool_completed_preview("x" * 5000, identity_redact)
        self.assertEqual(len(out), _TOOL_COMPLETED_PREVIEW_MAX_CHARS)
        self.assertTrue(out.endswith("..."))

    def test_redaction_runs_before_truncation(self):
        seen = {}

        def redact(text, force=False):
            seen["force"] = force
            return text.replace("SECRET", "[redacted]")

        out = _tool_completed_preview("SECRET value", redact)
        self.assertTrue(seen["force"])
        self.assertNotIn("SECRET", out)


if __name__ == "__main__":
    unittest.main()
