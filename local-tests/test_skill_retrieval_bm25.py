"""Tests of the opt-in per-turn BM25 skill-retrieval port (fork finding #118, lane c).

The patch ranks the loadable skills against the current user message with the BM25 scorer shared
with tools.tool_search_catalog (name, description, frontmatter triggers and tags) and appends a
short top-k hint to the user message's ``api_content`` sidecar only. It is off by default
(``skills.retrieval.enabled`` = False), ``top_k`` defaults to 3 and clamps to 1..5, settings are
resolved once at agent init (no per-turn load_config), ``skill_manage`` create/patch/delete
invalidates the corpus, and the system-prompt skill index is untouched.

The tests show, against real agents and real SKILL.md files in a temp HERMES_HOME:
 - enabled: "deploy a stack on the homelab" puts the homelab skill first among at most top_k hits;
 - disabled and default: ``api_content`` is unchanged and the stored content never grows;
 - the system prompt bytes are equal with retrieval enabled and disabled;
 - HERMES_SKILL_RETRIEVAL=on|off applies only when the config key is absent; config wins;
 - top_k 9 is clamped to 5, both in ranking and in the resolved settings;
 - CONFIG_SCHEMA exposes skills.retrieval.enabled and skills.retrieval.top_k;
 - the gateway env bridge exports the config values (config wins over a pre-set env).
"""
import os
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.skill_retrieval import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    clamp_top_k,
    retrieve_skills,
    resolve_retrieval_settings,
)
from hermes_cli.web_server_config import CONFIG_SCHEMA

QUERY = "deploy a stack on the homelab"

SKILLS = (
    # name, description, triggers, tags
    ("homelab-deploy", "Deploy a compose stack on the private homelab host.",
     ["deploy", "homelab", "stack"], ["ops", "docker"]),
    ("recipe-book", "Collect cooking recipes and convert between kitchen units.",
     [], ["kitchen"]),
    ("poetry-corner", "Write short poems and rhymes for greetings cards.",
     [], ["writing"]),
    ("weather-watch", "Read weather forecasts and answer daily weather questions.",
     [], ["weather"]),
)


def _skill_md(name, description, triggers, tags):
    lines = ["---", f"name: {name}", f"description: {description}"]
    if triggers:
        lines.append("triggers: [" + ", ".join(triggers) + "]")
    if tags:
        lines.append("tags: [" + ", ".join(tags) + "]")
    lines += ["---", f"Body of {name}.", ""]
    return "\n".join(lines)


class _FakeTodoStore:
    def has_items(self):
        return True


class _FakeGuardrails:
    def reset_for_turn(self):
        pass


class TurnAgent:
    """Minimal agent covering what the turn prologue touches (mirrors
    tests/agent/test_api_content_sidecar.py), plus the retrieval settings snapshot that
    ``agent.skill_retrieval.apply_retrieval_settings`` sets at init."""

    def __init__(self, cfg=None, system_prompt="SYSTEM"):
        from agent.skill_retrieval import apply_retrieval_settings

        self.session_id = "sess-1"
        self.model = "test/model"
        self.provider = "openrouter"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = "sk-x"
        self.api_mode = "chat_completions"
        self.platform = "cli"
        self.quiet_mode = True
        self.max_iterations = 90
        self.tools = []
        self.valid_tool_names = set()
        self._skip_mcp_refresh = True
        self.compression_enabled = False
        self.context_compressor = types.SimpleNamespace(protect_first_n=2, protect_last_n=2)
        self._cached_system_prompt = system_prompt
        self._memory_store = None
        self._memory_manager = None
        self._memory_nudge_interval = 0
        self._turns_since_memory = 0
        self._user_turn_count = 0
        self._todo_store = _FakeTodoStore()
        self._tool_guardrails = _FakeGuardrails()
        self._compression_warning = None
        self._interrupt_requested = False
        self._memory_write_origin = "assistant_tool"
        self._stream_context_scrubber = None
        self._stream_think_scrubber = None
        self.api_content_at_persist = "<unset>"
        apply_retrieval_settings(self, cfg if cfg is not None else {})

    def _ensure_db_session(self):
        pass

    def _restore_primary_runtime(self):
        pass

    def _cleanup_dead_connections(self):
        return False

    def _emit_status(self, _msg):
        pass

    def _replay_compression_warning(self):
        pass

    def _hydrate_todo_store(self, *_a, **_k):
        pass

    def _safe_print(self, *_a, **_k):
        pass

    def _persist_session(self, messages, _history=None):
        self.api_content_at_persist = messages[-1].get("api_content")


def _run_turn(agent, message=QUERY):
    from agent.turn_context import build_turn_context

    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        ctx = build_turn_context(
            agent=agent, user_message=message, system_message=None,
            conversation_history=None, task_id=None, stream_callback=None,
            persist_user_message=None,
            restore_or_build_system_prompt=lambda *a, **k: None,
            install_safe_stdio=lambda: None,
            sanitize_surrogates=lambda s: s,
            summarize_user_message_for_log=lambda s: s,
            set_session_context=lambda _sid: None,
            set_current_write_origin=lambda _o: None,
            ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
        )
    msg = ctx.messages[ctx.current_turn_user_idx]
    return msg.get("content"), ctx.active_system_prompt, msg


def _enabled_cfg(top_k=3):
    return {"skills": {"retrieval": {"enabled": True, "top_k": top_k}}}


class SkillRetrievalTest(unittest.TestCase):
    """One temp HERMES_HOME per test, with the four skills above and nothing under ~/.hermes/."""

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="hermes-skill-retrieval-")
        self._old_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = self.home
        self._old_env = {key: os.environ.get(key)
                         for key in ("HERMES_SKILL_RETRIEVAL", "HERMES_SKILL_RETRIEVAL_TOP_K")}
        os.environ.pop("HERMES_SKILL_RETRIEVAL", None)
        os.environ.pop("HERMES_SKILL_RETRIEVAL_TOP_K", None)
        for name, desc, triggers, tags in SKILLS:
            skill_dir = Path(self.home, "skills", "misc", name)
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(_skill_md(name, desc, triggers, tags),
                                                encoding="utf-8")

    def tearDown(self):
        from agent.skill_retrieval import invalidate_retrieval_index

        invalidate_retrieval_index()
        if self._old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self._old_home
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self.home, ignore_errors=True)

    def _hint_lines(self, api_content):
        return [line for line in api_content.splitlines() if line.startswith("- ")]

    def test_enabled_retrieves_matching_skill_first_within_top_k(self):
        content, _prompt, msg = _run_turn(TurnAgent(_enabled_cfg(top_k=2)))
        self.assertEqual(content, QUERY)  # stored content is untouched
        api_content = msg.get("api_content")
        self.assertIsNotNone(api_content)
        self.assertTrue(api_content.startswith(QUERY))
        hits = self._hint_lines(api_content)
        self.assertGreaterEqual(len(hits), 1)
        self.assertLessEqual(len(hits), 2)
        self.assertTrue(hits[0].startswith("- homelab-deploy:"), hits)
        self.assertIn("compose stack", hits[0])

    def test_disabled_is_off(self):
        _content, _prompt, msg = _run_turn(
            TurnAgent({"skills": {"retrieval": {"enabled": False, "top_k": 3}}}))
        self.assertIsNone(msg.get("api_content"))

    def test_default_config_is_off(self):
        from hermes_cli.config import DEFAULT_CONFIG

        content, _prompt, msg = _run_turn(TurnAgent(DEFAULT_CONFIG))
        self.assertEqual(content, QUERY)
        self.assertIsNone(msg.get("api_content"))

    def test_system_prompt_bytes_identical_enabled_and_disabled(self):
        _c1, enabled_prompt, _m1 = _run_turn(TurnAgent(_enabled_cfg()))
        _c2, disabled_prompt, _m2 = _run_turn(TurnAgent({}))
        self.assertIsInstance(enabled_prompt, str)
        self.assertEqual(enabled_prompt, disabled_prompt)

    def test_env_fallback_applies_only_when_config_key_absent(self):
        os.environ["HERMES_SKILL_RETRIEVAL"] = "on"
        self.assertEqual(resolve_retrieval_settings({}), (True, DEFAULT_TOP_K))
        os.environ["HERMES_SKILL_RETRIEVAL"] = "off"
        self.assertEqual(resolve_retrieval_settings({}), (False, DEFAULT_TOP_K))
        os.environ["HERMES_SKILL_RETRIEVAL"] = "on"
        self.assertEqual(resolve_retrieval_settings({"skills": {"retrieval": {"enabled": False}}}),
                         (False, DEFAULT_TOP_K))
        self.assertEqual(resolve_retrieval_settings(_enabled_cfg()), (True, DEFAULT_TOP_K))

    def test_env_on_end_to_end_turn_gets_hint(self):
        os.environ["HERMES_SKILL_RETRIEVAL"] = "on"
        os.environ["HERMES_SKILL_RETRIEVAL_TOP_K"] = "1"
        _content, _prompt, msg = _run_turn(TurnAgent({}))
        hits = self._hint_lines(msg["api_content"])
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0].startswith("- homelab-deploy:"), hits)

    def test_config_wins_over_env_for_top_k(self):
        os.environ["HERMES_SKILL_RETRIEVAL_TOP_K"] = "1"
        self.assertEqual(resolve_retrieval_settings(_enabled_cfg(top_k=4)), (True, 4))
        self.assertEqual(resolve_retrieval_settings({}), (False, 1))

    def test_top_k_nine_clamps_to_five_in_ranking_and_settings(self):
        self.assertEqual(clamp_top_k(9), MAX_TOP_K)
        self.assertEqual(resolve_retrieval_settings(_enabled_cfg(top_k=9)), (True, MAX_TOP_K))
        hits = retrieve_skills(QUERY, 9)
        self.assertLessEqual(len(hits), MAX_TOP_K)
        self.assertEqual(hits[0].name, "homelab-deploy")

    def test_trigger_and_tag_frontmatter_fields_participate(self):
        # None of these words occur in the name/description of poetry-corner; only its tag does.
        hits = retrieve_skills("writing", DEFAULT_TOP_K)
        self.assertEqual(hits[0].name, "poetry-corner")

    def test_config_schema_contains_both_keys(self):
        enabled = CONFIG_SCHEMA["skills.retrieval.enabled"]
        top_k = CONFIG_SCHEMA["skills.retrieval.top_k"]
        self.assertEqual(enabled["type"], "boolean")
        self.assertEqual(top_k["type"], "number")
        print("skills.retrieval.enabled:", enabled)
        print("skills.retrieval.top_k:", top_k)

    def test_gateway_bridge_exports_config_values_config_wins(self):
        from gateway.run import _bridge_skills_retrieval_to_env

        os.environ["HERMES_SKILL_RETRIEVAL"] = "off"
        os.environ["HERMES_SKILL_RETRIEVAL_TOP_K"] = "1"
        _bridge_skills_retrieval_to_env({"retrieval": {"enabled": True, "top_k": 4}})
        self.assertEqual(os.environ["HERMES_SKILL_RETRIEVAL"], "on")
        self.assertEqual(os.environ["HERMES_SKILL_RETRIEVAL_TOP_K"], "4")
        # Absent config key: pre-existing env stays untouched (fallback).
        _bridge_skills_retrieval_to_env({})
        self.assertEqual(os.environ["HERMES_SKILL_RETRIEVAL"], "on")

    def test_skill_manage_create_patch_delete_invalidates_index(self):
        agent = TurnAgent(_enabled_cfg(top_k=1))
        _content, _prompt, msg = _run_turn(agent)
        self.assertIn("homelab-deploy", msg["api_content"])
        from tools.skill_manager_tool import skill_manage
        from agent.skill_retrieval import get_index

        created = skill_manage(
            action="create", name="homelab-runner",
            content="---\nname: homelab-runner\ndescription: Deploy a stack on the homelab host.\n---\nBody.\n")
        self.assertIn('"success": true', created)
        self.assertEqual(retrieve_skills(QUERY, 1)[0].name, "homelab-runner")

        patched = skill_manage(
            action="patch", name="homelab-runner",
            old_string="Deploy a stack on the homelab host.",
            new_string="Crochet wool blankets for cold nights.")
        self.assertIn('"success": true', patched)
        self.assertEqual(retrieve_skills(QUERY, 1)[0].name, "homelab-deploy")

        deleted = skill_manage(action="delete", name="homelab-runner")
        self.assertIn('"success": true', deleted)
        self.assertNotIn("homelab-runner", [name for name, _desc in get_index().entries])
        self.assertEqual(retrieve_skills(QUERY, 1)[0].name, "homelab-deploy")


if __name__ == "__main__":
    unittest.main()
