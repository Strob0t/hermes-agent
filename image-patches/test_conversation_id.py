"""Tests of the local conversation-id patch (patch_conversation_id.py), run in the image build.

The patch hands the compression-stable ``conversation_id`` to the ``pre_stop`` and
``transform_llm_output`` hooks, so a plugin can keep per-conversation state across a compaction that
rotates ``session_id``. The tests show:
 - ``get_pre_stop_continue_message`` forwards ``conversation_id`` to the hook;
 - the real ``_pre_stop_nudge`` gate passes the agent's conversation root (not its session id);
 - the real ``apply_llm_output_transform`` puts ``conversation_id`` in the hook payload;
 - a failed lookup gives an empty string and never raises into the turn.
"""
import importlib.util
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import agent.turn_finalizer as tf
import agent.turn_stop_gates as tsg
from hermes_cli import plugins as plugins_facade


class RotatedAgent(SimpleNamespace):
    """An agent after a compaction: a new session_id, the same conversation root."""


def rotated_agent():
    return RotatedAgent(session_id="child-sess", model="m", platform="local",
                        _inherited_conversation_id="conv-root")


class PreStopDispatcherTest(unittest.TestCase):
    def test_conversation_id_reaches_the_hook(self):
        seen = {}

        def fake_invoke(name, **kw):
            seen.update(kw)
            return [{"action": "continue", "message": "go on"}]

        with patch.object(plugins_facade, "invoke_hook", fake_invoke):
            msg = plugins_facade.get_pre_stop_continue_message(
                session_id="child-sess", platform="local", attempt=0, final_response="x",
                conversation_id="conv-root")
        self.assertEqual(msg, "go on")
        self.assertEqual(seen["conversation_id"], "conv-root")
        self.assertEqual(seen["session_id"], "child-sess")  # the old fields are unchanged

    def test_conversation_id_is_optional(self):
        seen = {}

        def fake_invoke(name, **kw):
            seen.update(kw)
            return []

        with patch.object(plugins_facade, "invoke_hook", fake_invoke):
            self.assertIsNone(plugins_facade.get_pre_stop_continue_message(session_id="s"))
        self.assertEqual(seen["conversation_id"], "")


class GateTest(unittest.TestCase):
    def test_gate_passes_the_conversation_root_not_the_session_id(self):
        captured = {}

        def fake_msg(**kw):
            captured.update(kw)
            return "continue please"

        with patch("hermes_cli.lifecycle.has_hook", return_value=True), \
             patch.object(plugins_facade, "get_pre_stop_continue_message", fake_msg):
            out = tsg._pre_stop_nudge(rotated_agent(), "answer", 0)
        self.assertEqual(out, "continue please")
        self.assertEqual(captured["conversation_id"], "conv-root")
        self.assertEqual(captured["session_id"], "child-sess")

    def test_failed_lookup_gives_an_empty_string(self):
        with patch("agent.prompt_cache_scope.conversation_id", side_effect=RuntimeError("boom")):
            self.assertEqual(tsg._conversation_id_safe(rotated_agent()), "")


class TransformPayloadTest(unittest.TestCase):
    def test_payload_carries_conversation_id(self):
        seen = {}

        def fake_invoke(name, logger, **kw):
            seen.update(kw)
            return []

        agent = rotated_agent()
        with patch.object(tf, "_invoke_hook_safely", fake_invoke):
            tf.apply_llm_output_transform(agent, "MODEL REPORT", turn_id="t1", logger=SimpleNamespace())
        self.assertEqual(seen["conversation_id"], "conv-root")
        self.assertEqual(seen["session_id"], "child-sess")
        self.assertEqual(seen["response_text"], "MODEL REPORT")




def _load_patcher():
    """Import patch_conversation_id.py as a module so the test uses its real PATCHES + guard logic;
    importing does not run main() (guarded by __main__). Returns None when the patcher is not next
    to the test: the image build copies only the test file into local-tests/, so these guard tests
    skip there (the build already proves the patch applies) and run in a dev checkout where the
    patcher is adjacent. When the Dockerfile copies the patcher alongside, they run at build time."""
    for cand in (os.path.join(os.path.dirname(__file__), "patch_conversation_id.py"),
                 os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "image-patches", "patch_conversation_id.py")):
        if os.path.isfile(cand):
            spec = importlib.util.spec_from_file_location("patch_conversation_id", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


# The exact text (pristine v0.23.1 hermes_cli/plugins.py, ~line 1119) that made the OLD guard
# false-match a clean tree: an unrelated dispatcher already carries `conversation_id: str = ""`.
_DECOY = (
    "    middleware_trace: Optional[list[dict[str, Any]]] = None, conversation_id: str = \"\",\n"
)


class PatcherGuardUnitTest(unittest.TestCase):
    """F10: already_patched() must treat the pristine decoy as NOT patched, and this patch's own
    output as patched."""

    def setUp(self):
        self.mod = _load_patcher()
        if self.mod is None:
            self.skipTest("patch_conversation_id.py not next to the test (build harness)")

    def test_decoy_is_not_treated_as_patched(self):
        self.assertFalse(self.mod.already_patched("hermes_cli/plugins.py", _DECOY))

    def test_own_output_is_treated_as_patched(self):
        patched = "final_response=final_response, conversation_id=conversation_id,"
        self.assertTrue(self.mod.already_patched("hermes_cli/plugins.py", patched))

    def test_helper_file_marker_matches(self):
        self.assertTrue(self.mod.already_patched("agent/turn_stop_gates.py",
                                                 "def _conversation_id_safe(agent):"))


class PatcherEndToEndTest(unittest.TestCase):
    """F10 end-to-end: a tree assembled from the patcher's OWN anchors (so they match by
    construction) plus the decoy line must apply once and then abort on a re-run."""

    def setUp(self):
        self.mod = _load_patcher()
        if self.mod is None:
            self.skipTest("patch_conversation_id.py not next to the test (build harness)")

    def _make_tree(self):
        root = tempfile.mkdtemp(prefix="f10-e2e-")
        per_file = {}
        for rel, old, _new in self.mod.PATCHES:
            per_file.setdefault(rel, []).append(old)
        for rel, olds in per_file.items():
            os.makedirs(os.path.join(root, os.path.dirname(rel)), exist_ok=True)
            body = "\n\n\n".join(olds)
            if rel == "hermes_cli/plugins.py":
                body = _DECOY + "\n\n" + body  # inject the pristine decoy
            with open(os.path.join(root, rel), "w", encoding="utf-8") as f:
                f.write(body + "\n")
        return root

    def test_apply_once_then_abort_on_rerun(self):
        root = self._make_tree()
        # first apply: patch main() against ROOT=root; it should write all files and not exit
        self.mod.ROOT = root
        self.mod.main()  # raises SystemExit only on failure
        plugins = open(os.path.join(root, "hermes_cli", "plugins.py"), encoding="utf-8").read()
        self.assertIn("final_response=final_response, conversation_id=conversation_id,", plugins)
        self.assertIn("conversation_id: str = \"\",", plugins)  # decoy still present, untouched extra
        gates = open(os.path.join(root, "agent", "turn_stop_gates.py"), encoding="utf-8").read()
        self.assertIn("_conversation_id_safe", gates)
        # re-run must abort as already patched
        with self.assertRaises(SystemExit) as cm:
            self.mod.main()
        self.assertIn("already patched", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
