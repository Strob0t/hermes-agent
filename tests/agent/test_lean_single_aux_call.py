"""Lean compaction makes EXACTLY ONE auxiliary LLM request per attempt.

Contract (#96603 — the per-chunk digest loop made up to 28 extra aux calls
and pushed compactions to 7-11 minutes on slow aux routes):

(a) exactly one ``call_llm`` per compaction attempt in lean mode;
(b) the single response's detailed-session-log section lands in the summary;
(c) an oversized region is EVEN-SAMPLED into the one request's input (with
    explicit elision markers), never split into extra requests;
(d) the LLM-free anchor index and the session_search recovery footer are
    still appended.
"""

from unittest.mock import patch, MagicMock

import pytest

from agent.context_compressor import (
    ContextCompressor,
    _LEAN_ANCHOR_HEADING,
    _LEAN_RECOVERY_HEADING,
    _LEAN_SESSION_LOG_HEADING,
)


def _mk_compressor(**overrides):
    kwargs = dict(
        model="test/model",
        threshold_percent=0.85,
        protect_first_n=2,
        protect_last_n=2,
        quiet_mode=True,
        tail_mode="lean",
    )
    kwargs.update(overrides)
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100_000
    ):
        c = ContextCompressor(**kwargs)
        _ = c.context_length
    c._session_id = "sess-lean-1call"
    return c


def _llm_response(text):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = MagicMock()
    resp.choices[0].message.content = text
    return resp


def _call_prompts(mock_call):
    return ["\n".join(m["content"] for m in call.kwargs["messages"]) for call in mock_call.call_args_list]


def _assert_each_round_in_one_pass(prompts, n_rounds):
    for i in range(n_rounds):
        assert sum(f"round {i} output: " in prompt for prompt in prompts) == 1, i


def _big_region(n_rounds=60, tool_chars=6_000):
    """Synthetic compacted region: user + assistant/tool rounds, ~360K chars."""
    turns = [
        {"role": "user", "content": "Please fix PR #12345 in agent/foo.py"},
    ]
    for i in range(n_rounds):
        turns.append({
            "role": "assistant",
            "content": f"Working on step {i}: editing agent/foo.py line {i}",
        })
        turns.append({
            "role": "tool",
            "tool_call_id": f"tc{i}",
            "tool_name": "terminal",
            "content": f"round {i} output: " + ("x" * tool_chars),
        })
    return turns


SUMMARY_BODY = (
    "## Historical Task Snapshot\nUser asked: 'Please fix PR #12345 in agent/foo.py'\n\n"
    "## Goal\nFix the bug.\n\n"
    "## Completed Actions\n1. EDIT agent/foo.py — done [tool: patch]\n\n"
    f"{_LEAN_SESSION_LOG_HEADING}\n- Edited agent/foo.py; PR #12345; ran pytest.\n"
)


class TestLeanSingleAuxiliaryCall:
    def test_exactly_one_call_llm_per_lean_attempt(self):
        """Lean input below the bound makes exactly one auxiliary request; input over the bound
        makes more than one, and every round reaches exactly one of them."""
        c = _mk_compressor()
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_llm_response(SUMMARY_BODY),
        ) as main_call, patch(
            "agent.auxiliary_client.call_llm",
            return_value=_llm_response(SUMMARY_BODY),
        ) as aux_call:
            summary = c._generate_summary(_big_region(n_rounds=5))
        assert summary is not None
        assert main_call.call_count + aux_call.call_count == 1

        turns = _big_region()
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_llm_response(SUMMARY_BODY),
        ) as main_call, patch(
            "agent.auxiliary_client.call_llm",
            return_value=_llm_response(SUMMARY_BODY),
        ) as aux_call:
            summary = c._generate_summary(turns)
        assert summary is not None
        assert aux_call.call_count == 0
        assert main_call.call_count > 1
        _assert_each_round_in_one_pass(_call_prompts(main_call), 60)

    def test_session_log_heading_lands_in_summary(self):
        c = _mk_compressor()
        turns = _big_region(n_rounds=5)
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_llm_response(SUMMARY_BODY),
        ):
            summary = c._generate_summary(turns)
        assert _LEAN_SESSION_LOG_HEADING in summary


    def test_anchor_index_and_recovery_footer_present(self):
        c = _mk_compressor()
        turns = _big_region(n_rounds=5)
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_llm_response(SUMMARY_BODY),
        ):
            summary = c._generate_summary(turns)
        assert _LEAN_ANCHOR_HEADING in summary
        assert _LEAN_RECOVERY_HEADING in summary
        assert "session_search" in summary

    def test_oversized_region_sampled_not_split_into_more_requests(self):
        """An oversized lean region is split into several bounded requests in order, with no
        elided middle: every round reaches exactly one request."""
        c = _mk_compressor()
        turns = _big_region(n_rounds=120, tool_chars=6_000)  # ~720K chars raw
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_llm_response(SUMMARY_BODY),
        ) as mock_call:
            c._generate_summary(turns)
        assert mock_call.call_count > 1
        prompts = _call_prompts(mock_call)
        for prompt in prompts:
            assert len(prompt) <= c._SUMMARY_INPUT_MAX_CHARS + 20_000
            assert "chars elided" not in prompt
        _assert_each_round_in_one_pass(prompts, 120)
        first_pass = [min(n for n, prompt in enumerate(prompts) if f"round {i} output: " in prompt) for i in range(120)]
        assert first_pass == sorted(first_pass)


class TestSampledSummaryInput:
    def test_small_input_passes_through(self):
        records = ["[USER]: abc" * 100] * 3
        assert ContextCompressor._sample_summary_records(records)[0] == "\n\n".join(records)

    def test_sampling_is_bounded_ordered_and_marked(self):
        # Distinct decade markers let us verify oldest-to-newest order and
        # uniform coverage across the whole region.
        records = [f"<seg{i:02d}>" + ("x" * 50_000) for i in range(10)]
        out, _coverage = ContextCompressor._sample_summary_records(records)
        assert len(out) <= ContextCompressor._SUMMARY_INPUT_MAX_CHARS
        assert "chars elided" in out
        seen = [i for i in range(10) if f"<seg{i:02d}>" in out]
        # Coverage reaches past the head AND includes the newest end.
        assert seen == sorted(seen)
        assert any(i >= 5 for i in seen)
        # The newest record anchors the last slice, so its token must be present outright.
        assert "<seg09>" in out

    def test_legacy_mode_keeps_head_tail_bound(self):
        """Legacy mode also splits oversized input into several passes, so no pass needs the
        head/tail truncation and every round reaches exactly one pass. Input below the bound
        stays one call."""
        c = _mk_compressor(tail_mode="legacy")
        turns = _big_region(n_rounds=120, tool_chars=6_000)
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_llm_response("## Historical Task Snapshot\nNone."),
        ) as mock_call:
            c._generate_summary(turns)
        assert mock_call.call_count > 1
        prompts = _call_prompts(mock_call)
        for prompt in prompts:
            assert "summary input truncated" not in prompt
        _assert_each_round_in_one_pass(prompts, 120)

        with patch(
            "agent.context_compressor.call_llm",
            return_value=_llm_response("## Historical Task Snapshot\nNone."),
        ) as small_call:
            c._generate_summary(_big_region(n_rounds=5))
        assert small_call.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
