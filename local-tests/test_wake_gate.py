"""Tests of the local wake-gate patch, run in the image build.

The patch gives plugins two facts that the goal-loop go rule needs:
 - ``pre_llm_call`` gets the compression-stable ``conversation_id`` (agent/turn_context.py), so a
   plugin can key a per-conversation gate that survives a compaction that rotates ``session_id``;
 - a gateway wake self-post over ``/v1/chat/completions`` (gateway/wake.py) is marked
   ``hermes_wake``; the admission carries ``wake_turn`` and session_ingress then does not tag the
   user row ``api_inbound``, so the wake is not read as a client message.
"""
import asyncio
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import agent.turn_context as tc
import gateway.session_api_turn as sat
import gateway.session_ingress as si
import gateway.wake as wake


def rotated_agent():
    return SimpleNamespace(session_id="child-sess", model="m", platform="api_server",
                           _inherited_conversation_id="conv-root", _persist_disabled=False)


def collect(agent):
    seen = {}

    def fake_invoke(name, **kw):
        seen["name"] = name
        seen.update(kw)
        return []

    with patch("hermes_cli.lifecycle.invoke_hook", fake_invoke):
        out = tc._collect_pre_llm_call_context(
            agent, effective_task_id="t", turn_id="turn", original_user_message="go",
            messages=[{"role": "user", "content": "go"}], conversation_history=[])
    return out, seen


class PreLlmCallConversationIdTest(unittest.TestCase):
    def test_conversation_id_reaches_the_hook(self):
        out, seen = collect(rotated_agent())
        self.assertEqual(out, "")
        self.assertEqual(seen["name"], "pre_llm_call")
        self.assertEqual(seen["conversation_id"], "conv-root")
        self.assertEqual(seen["session_id"], "child-sess")  # the old fields are unchanged

    def test_failed_lookup_gives_an_empty_string_and_the_hook_still_runs(self):
        with patch("agent.prompt_cache_scope.conversation_id", side_effect=RuntimeError("boom")):
            _, seen = collect(rotated_agent())
        self.assertEqual(seen["name"], "pre_llm_call")
        self.assertEqual(seen["conversation_id"], "")


class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def read(self):
        return b"{}"

    async def text(self):
        return "{}"


class _FakeSession:
    posts = []

    def __init__(self, **_):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, json=None, headers=None):
        _FakeSession.posts.append({"url": url, "json": json, "headers": headers})
        return _FakeResponse()


class WakeSelfPostTest(unittest.TestCase):
    def test_self_post_marks_the_turn_as_a_wake(self):
        fake = types.ModuleType("aiohttp")
        fake.ClientTimeout = lambda **kw: None
        fake.ClientSession = _FakeSession
        fake.ClientError = OSError
        _FakeSession.posts = []
        adapter = SimpleNamespace(_host="127.0.0.1", _port=8642, _api_key="k", _model_name="hermes-agent")
        with patch.dict(sys.modules, {"aiohttp": fake}):
            asyncio.run(wake._self_post_chat_completion(adapter, text="[kanban] Task t1 completed.",
                                                        session_id="sess-1"))
        self.assertEqual(len(_FakeSession.posts), 1)
        body = _FakeSession.posts[0]["json"]
        self.assertIs(body["hermes_wake"], True)
        self.assertTrue(_FakeSession.posts[0]["url"].endswith("/v1/chat/completions"))
        self.assertEqual(_FakeSession.posts[0]["headers"]["X-Hermes-Session-Id"], "sess-1")


class AdmissionWakeTurnTest(unittest.TestCase):
    def payload(self, **extra):
        return {"text": "x", "api_turn_v1": {"history": None, "settings": {"route_source": "global"}, **extra}}

    def test_prepare_api_execution_carries_wake_turn(self):
        with patch.object(sat, "check_api_turn", lambda *a: "adapter"), \
             patch.object(sat, "api_settings", lambda *a: {}):
            self.assertTrue(sat.prepare_api_execution(None, None, self.payload(wake_turn=True))["wake_turn"])
            self.assertFalse(sat.prepare_api_execution(None, None, self.payload())["wake_turn"])
            self.assertFalse(sat.prepare_api_execution(None, None, {"text": "x"})["wake_turn"])

    def _authority(self):
        runner = SimpleNamespace()
        adapter = SimpleNamespace(gateway_runner=runner)
        runner._adapter_for_source = lambda source: adapter
        live = SimpleNamespace(source=SimpleNamespace(platform="not-api"))
        return SimpleNamespace(sessions={"s": live}, runner=runner), SimpleNamespace(session_id="s")

    def test_check_api_turn_accepts_true_and_refuses_other_values(self):
        authority, ref = self._authority()
        sat.check_api_turn(authority, ref, self.payload(wake_turn=True))
        sat.check_api_turn(authority, ref, self.payload())
        for bad in (False, "yes", 1):
            with self.assertRaises(sat.AuthorityRefusal):
                sat.check_api_turn(authority, ref, self.payload(wake_turn=bad))


class ApiInboundTagTest(unittest.TestCase):
    def test_only_a_client_turn_is_tagged(self):
        client = SimpleNamespace(internal=False)
        self.assertTrue(si._is_client_api_turn(client, {"wake_turn": False}))
        self.assertTrue(si._is_client_api_turn(client, None))
        self.assertFalse(si._is_client_api_turn(client, {"wake_turn": True}))
        self.assertFalse(si._is_client_api_turn(SimpleNamespace(internal=True), {"wake_turn": False}))


if __name__ == "__main__":
    unittest.main()
