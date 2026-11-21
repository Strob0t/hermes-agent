"""Tests of the local context-window patch (patch_context_window.py), run in the image build.

The compression trigger is a line over the window size, and the window of the compressor follows the model that
Bifrost names as the one that served the request.
"""
import types
import unittest
from unittest.mock import patch

from agent import served_model
from agent.context_compressor import ContextCompressor
from agent.model_metadata import MINIMUM_CONTEXT_LENGTH

CONFIGURED = 0.40
LIST = {"claude-sub/claude-sonnet-5": {"context_length": 1_000_000}, "groq/small": {"context_length": 131_072},
        "small": {"context_length": 131_072}, "cloudflare/no-window": {"name": "x"}}


def compressor(window, threshold=CONFIGURED):
    with patch("agent.context_compressor.get_model_context_length", return_value=window):
        built = ContextCompressor(model="tier-medium", quiet_mode=True, threshold_percent=threshold)
        _ = built.context_length
    return built


def agent_with(window=1_000_000):
    return types.SimpleNamespace(base_url="http://bifrost-relay:8080/v1", api_key="k", last_served_model=None,
                                 context_compressor=compressor(window))


def respond(agent, headers, status=200):
    client = types.SimpleNamespace(_client=types.SimpleNamespace(event_hooks={"response": []}))
    served_model.install_served_model_capture(agent, client)
    hook = client._client.event_hooks["response"][-1]
    with patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value=LIST) as fetch:
        hook(types.SimpleNamespace(status_code=status, headers=headers))
    return fetch


class TriggerCurveTest(unittest.TestCase):
    def percent(self, window, threshold=CONFIGURED):
        return ContextCompressor._effective_threshold_percent(window, threshold)

    def test_a_large_window_keeps_the_configured_threshold(self):
        self.assertEqual(self.percent(1_000_000), CONFIGURED)
        self.assertEqual(self.percent(1_048_576), CONFIGURED)

    def test_the_smallest_window_triggers_at_85_percent(self):
        self.assertEqual(self.percent(MINIMUM_CONTEXT_LENGTH), 0.85)
        self.assertEqual(self.percent(30_000), 0.85)

    def test_the_trigger_is_a_line_between_the_two_ends(self):
        middle = (1_000_000 + MINIMUM_CONTEXT_LENGTH) // 2
        self.assertAlmostEqual(self.percent(middle), 0.625, places=6)
        self.assertAlmostEqual(self.percent(256_000), 0.40 + 0.45 * (744_000 / 936_000), places=6)
        values = [self.percent(w) for w in (64_000, 131_072, 200_000, 256_000, 512_000, 999_999, 1_000_000)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_a_threshold_above_the_small_end_and_an_unknown_window_stay(self):
        self.assertEqual(self.percent(131_072, 0.90), 0.90)
        self.assertEqual(self.percent(0), CONFIGURED)

    def test_the_compressor_uses_the_curve_for_its_trigger(self):
        self.assertEqual(compressor(1_000_000).threshold_tokens, 400_000)
        small = compressor(131_072)
        self.assertAlmostEqual(small.threshold_percent, 0.40 + 0.45 * (868_928 / 936_000), places=6)
        self.assertEqual(small.threshold_tokens, int(131_072 * small.threshold_percent))


class CurveAndSmallWindowFloorTest(unittest.TestCase):
    def percent(self, window, threshold=CONFIGURED):
        return ContextCompressor._effective_threshold_percent(window, threshold)

    def curve(self, window, threshold=CONFIGURED):
        return threshold + (0.85 - threshold) * (1_000_000 - window) / (1_000_000 - MINIMUM_CONTEXT_LENGTH)

    def test_the_floor_applies_where_the_curve_is_below_75_percent(self):
        self.assertEqual(self.percent(400_000), 0.75)
        self.assertEqual(self.percent(300_000), 0.75)

    def test_the_curve_applies_where_it_is_above_the_floor(self):
        self.assertGreater(self.curve(256_000), 0.75)
        self.assertAlmostEqual(self.percent(256_000), self.curve(256_000), places=6)
        self.assertGreaterEqual(self.percent(400_000, 0.80), 0.80)
        self.assertAlmostEqual(self.percent(400_000, 0.80), self.curve(400_000, 0.80), places=6)

    def test_no_floor_at_or_above_the_limit(self):
        self.assertLess(self.curve(512_000), 0.75)
        self.assertAlmostEqual(self.percent(512_000), self.curve(512_000), places=6)
        self.assertEqual(self.percent(1_000_000), CONFIGURED)


class ServedModelTest(unittest.TestCase):
    def test_the_bifrost_headers_give_provider_and_model(self):
        self.assertEqual(served_model.served_model_from_headers(
            {"x-bifrost-provider": "claude-sub", "x-bifrost-resolved-model": "claude-sonnet-5"}),
            "claude-sub/claude-sonnet-5")
        self.assertEqual(served_model.served_model_from_headers(
            {"x-bifrost-provider": "groq", "x-bifrost-resolved-model": "openai/gpt-oss-20b"}), "groq/openai/gpt-oss-20b")
        self.assertEqual(served_model.served_model_from_headers({"x-bifrost-resolved-model": "m"}), "m")
        self.assertEqual(served_model.served_model_from_headers(
            {"x-bifrost-provider": "groq", "x-bifrost-resolved-model": "groq/m"}), "groq/m")
        self.assertIsNone(served_model.served_model_from_headers({"x-bifrost-provider": "groq"}))

    def test_a_litellm_header_wins(self):
        self.assertEqual(served_model.served_model_from_headers(
            {"x-litellm-model-id": "dep-1", "x-bifrost-provider": "groq", "x-bifrost-resolved-model": "m"}), "dep-1")

    def test_the_window_follows_the_served_model(self):
        agent = agent_with(1_000_000)
        respond(agent, {"x-bifrost-provider": "groq", "x-bifrost-resolved-model": "small"})
        self.assertEqual(agent.last_served_model, "groq/small")
        self.assertEqual(agent.context_compressor.context_length, 131_072)
        self.assertEqual(agent.context_compressor.threshold_tokens,
                         int(131_072 * ContextCompressor._effective_threshold_percent(131_072, CONFIGURED)))
        respond(agent, {"x-bifrost-provider": "claude-sub", "x-bifrost-resolved-model": "claude-sonnet-5"})
        self.assertEqual(agent.context_compressor.context_length, 1_000_000)
        self.assertEqual(agent.context_compressor.threshold_tokens, 400_000)

    def test_the_list_is_read_only_when_the_served_model_changes(self):
        agent = agent_with(1_000_000)
        headers = {"x-bifrost-provider": "groq", "x-bifrost-resolved-model": "small"}
        self.assertEqual(respond(agent, headers).call_count, 1)
        self.assertEqual(respond(agent, headers).call_count, 0)

    def test_an_unknown_model_a_model_without_a_window_and_an_error_answer_change_nothing(self):
        for headers, status in (({"x-bifrost-provider": "x", "x-bifrost-resolved-model": "unknown"}, 200),
                                ({"x-bifrost-provider": "cloudflare", "x-bifrost-resolved-model": "no-window"}, 200),
                                ({"x-bifrost-provider": "groq", "x-bifrost-resolved-model": "small"}, 400),
                                ({}, 200)):
            agent = agent_with(1_000_000)
            respond(agent, headers, status)
            self.assertEqual(agent.context_compressor.context_length, 1_000_000)

    def test_an_agent_without_a_compressor_or_a_base_url_does_not_fail(self):
        agent = types.SimpleNamespace(base_url="", api_key="", last_served_model=None, context_compressor=None)
        respond(agent, {"x-bifrost-provider": "groq", "x-bifrost-resolved-model": "small"})
        self.assertEqual(agent.last_served_model, "groq/small")


def session_agent(session_id="s1", key="k1", window=1_000_000):
    agent = agent_with(window)
    agent.session_id = session_id
    agent._gateway_session_key = key
    return agent


class SessionWindowTest(unittest.TestCase):
    """An agent is built for each turn: the checks before a turn use the window of the model that served the
    session last."""

    SMALL = {"x-bifrost-provider": "groq", "x-bifrost-resolved-model": "small"}
    LARGE = {"x-bifrost-provider": "claude-sub", "x-bifrost-resolved-model": "claude-sonnet-5"}

    def setUp(self):
        served_model._session_windows.clear()

    def test_the_served_window_is_kept_for_the_session_key_and_the_session_id(self):
        respond(session_agent(), self.SMALL)
        self.assertEqual(served_model.session_window("k1"), ("groq/small", 131_072))
        self.assertEqual(served_model.session_window("s1"), ("groq/small", 131_072))
        self.assertEqual(served_model.session_window(None, "s1"), ("groq/small", 131_072))
        self.assertIsNone(served_model.session_window("other"))
        self.assertIsNone(served_model.session_window(None, ""))

    def test_a_new_agent_of_the_session_starts_with_the_served_window(self):
        respond(session_agent(), self.SMALL)
        fresh = session_agent()
        self.assertEqual(fresh.context_compressor.threshold_tokens, 400_000)
        served_model.apply_session_window(fresh)
        self.assertEqual(fresh.context_compressor.context_length, 131_072)
        self.assertEqual(fresh.context_compressor.threshold_tokens,
                         int(131_072 * ContextCompressor._effective_threshold_percent(131_072, CONFIGURED)))

    def test_the_session_id_alone_finds_the_record(self):
        respond(session_agent(key=None), self.SMALL)
        fresh = session_agent(key=None)
        served_model.apply_session_window(fresh)
        self.assertEqual(fresh.context_compressor.context_length, 131_072)

    def test_the_record_follows_the_model_back_to_the_large_window(self):
        respond(session_agent(), self.SMALL)
        respond(session_agent(), self.LARGE)
        self.assertEqual(served_model.session_window("k1"), ("claude-sub/claude-sonnet-5", 1_000_000))
        fresh = session_agent()
        served_model.apply_session_window(fresh)
        self.assertEqual(fresh.context_compressor.context_length, 1_000_000)

    def test_a_different_session_no_record_and_an_old_record_change_nothing(self):
        respond(session_agent(), self.SMALL)
        other = session_agent(session_id="s2", key="k2")
        served_model.apply_session_window(other)
        self.assertEqual(other.context_compressor.context_length, 1_000_000)
        with patch.object(served_model, "SESSION_WINDOW_TTL", -1.0):
            fresh = session_agent()
            served_model.apply_session_window(fresh)
            self.assertEqual(fresh.context_compressor.context_length, 1_000_000)
            self.assertEqual(served_model.session_window_or(1_000_000, "k1", "s1"), 1_000_000)

    def test_an_unknown_model_and_an_error_answer_make_no_record(self):
        respond(session_agent(), {"x-bifrost-provider": "x", "x-bifrost-resolved-model": "unknown"})
        respond(session_agent(), self.SMALL, 400)
        self.assertIsNone(served_model.session_window("k1", "s1"))

    def test_an_agent_without_session_keys_or_compressor_does_not_fail(self):
        respond(agent_with(), self.SMALL)
        self.assertEqual(served_model._session_windows, {})
        served_model.apply_session_window(types.SimpleNamespace(context_compressor=None))
        served_model.apply_session_window(types.SimpleNamespace())

    def test_the_record_store_has_a_size_limit(self):
        with patch.object(served_model, "SESSION_WINDOW_MAX", 20):
            for n in range(60):
                respond(session_agent(session_id=f"s{n}", key=None), self.SMALL)
            self.assertLessEqual(len(served_model._session_windows), 20)
            self.assertEqual(served_model.session_window("s59"), ("groq/small", 131_072))

    def test_the_hygiene_value_is_the_served_window_or_the_default(self):
        respond(session_agent(), self.SMALL)
        self.assertEqual(served_model.session_window_or(1_000_000, "k1", None), 131_072)
        self.assertEqual(served_model.session_window_or(1_000_000, None, "s1"), 131_072)
        self.assertEqual(served_model.session_window_or(1_000_000, "other", None), 1_000_000)

    def test_the_compaction_at_the_start_of_a_turn_uses_the_served_window(self):
        from agent import turn_context_compaction as compaction
        respond(session_agent(), self.SMALL)
        fresh = session_agent()
        fresh.compression_enabled = False  # the turn-start cold prune is not under test
        seen = []
        with patch.object(compaction, "_idle_compaction",
                          side_effect=lambda agent, *a: seen.append(agent.context_compressor.context_length)), \
                patch.object(compaction, "_preflight_compression",
                             side_effect=lambda agent, *a: seen.append(agent.context_compressor.threshold_tokens)):
            compaction.run_turn_start_compaction(
                fresh, messages=[], system_message=None, active_system_prompt=None, conversation_history=None,
                current_turn_user_idx=0, user_message="x", effective_task_id="t")
        self.assertEqual(seen, [131_072, fresh.context_compressor.threshold_tokens])
        self.assertLess(seen[1], 131_072)


class HygieneCheckTest(unittest.IsolatedAsyncioTestCase):
    """The session hygiene of the gateway (85 % of the window) uses the window of the served model."""

    def setUp(self):
        served_model._session_windows.clear()

    async def plan(self, tokens):
        from gateway.run_turn_hygiene import GatewayTurnHygieneMixin as Mixin

        async def not_in_flight(_key):
            return False

        async def alias_window(*_a, **_k):
            return 1_000_000

        runner = types.SimpleNamespace(_HygienePlan=Mixin._HygienePlan, _session_db=None,
                                       _session_has_compression_in_flight=not_in_flight)
        settings = types.SimpleNamespace(model="tier-medium", base_url="http://bifrost-relay:8080/v1", api_key="k",
                                         config_context_length=None, provider="custom", threshold_pct=0.85,
                                         hard_msg_limit=5000)
        entry = types.SimpleNamespace(session_id="s1", last_prompt_tokens=tokens)
        history = [{"role": "user", "content": "x"}] * 6
        with patch("agent.model_metadata.get_model_context_length_async", side_effect=alias_window):
            return await Mixin._hmwa_hygiene_plan(runner, settings, history, entry, "k1")

    async def test_without_a_record_the_alias_window_applies(self):
        self.assertFalse((await self.plan(200_000)).needs_compress)
        self.assertTrue((await self.plan(860_000)).needs_compress)

    async def test_with_a_record_the_served_window_applies(self):
        respond(session_agent(), SessionWindowTest.SMALL)
        self.assertTrue((await self.plan(200_000)).needs_compress)
        self.assertFalse((await self.plan(100_000)).needs_compress)
        plan = await self.plan(200_000)
        self.assertEqual(plan.warn_token_threshold, int(131_072 * 0.95))


if __name__ == "__main__":
    unittest.main()
