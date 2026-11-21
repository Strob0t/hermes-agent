"""A stale current-turn index must not move the persist override onto a later user row of the same turn.

The flush writes ``_persist_user_message_override`` (the clean text of this turn's ask) onto the row at
``_persist_user_message_idx``. After the first flush the live row can hold that durable text, while
``user_message`` keeps the wire text (for example with a gateway timestamp). The iteration prep then
found no row with the wire text, re-anchored to the newest user-originated row, and that row was a
``pre_stop`` continuation nudge or a ``/steer`` row. The flush wrote it with the owner's text: the
nudge and the steer were lost from state.db and the ask showed up again as a new active row on each
re-anchor ("Re-anchored stale current_turn_user_idx X -> Y" with X != Y).
"""

import pytest

ASK = "check the open points against the live state"
WIRE = "[2026-11-20T14:30:02+0100] " + ASK
NUDGE = "[goal-loop] 1 goal(s) are still open. Continue the work on these goals."
STEER = "A (recommended): keep the workbench path"


def _agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    from hermes_state import SessionDB

    return AIAgent(session_db=SessionDB(db_path=tmp_path / "proof.db"),
                   model="test-model", provider="openai-compat", api_key="test",
                   base_url="http://127.0.0.1:1/v1", max_iterations=4,
                   quiet_mode=True, skip_context_files=True, skip_memory=True)


def _later_row(kind):
    if kind == "pre_stop":
        return {"role": "user", "content": NUDGE, "_pre_stop_synthetic": True}
    from agent.prompt_builder import steer_user_row

    return steer_user_row(STEER)


def _user_rows(agent):
    return [m.get("content") for m in agent._session_db.get_messages(agent.session_id) if m.get("role") == "user"]


@pytest.mark.parametrize("kind", ["pre_stop", "steer"])
def test_iteration_prep_keeps_the_anchor_on_the_flushed_turn_row(tmp_path, monkeypatch, kind):
    from agent.turn_context import _reset_per_turn_agent_state
    from agent.turn_iteration_prep import prepare_iteration

    agent = _agent(tmp_path, monkeypatch)
    try:
        _reset_per_turn_agent_state(agent)
        agent._persist_user_message_override = ASK
        # The state after the turn-start flush: the live row holds the durable text, not the wire text.
        messages = [
            {"role": "user", "content": "old ask"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": ASK},
            {"role": "assistant", "content": "interim answer", "finish_reason": "stop_hook_continue"},
            _later_row(kind),
        ]
        later = messages[4]["content"]
        agent._persist_user_message_idx = 2
        prep = prepare_iteration(agent, messages=messages, api_call_count=2, user_message=WIRE,
                                 current_turn_user_idx=2)
        assert prep.current_turn_user_idx == 2
        assert agent._persist_user_message_idx == 2
        agent._flush_messages_to_session_db(prep.messages, conversation_history=[])
        assert _user_rows(agent) == ["old ask", ASK, later]
    finally:
        agent._session_db.close()


@pytest.mark.parametrize("kind", ["pre_stop", "steer"])
def test_flush_never_writes_the_override_onto_a_nudge_or_steer_row(tmp_path, monkeypatch, kind):
    agent = _agent(tmp_path, monkeypatch)
    try:
        agent._persist_user_message_override = ASK
        messages = [
            {"role": "user", "content": "old ask"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": ASK},
            {"role": "assistant", "content": "interim answer", "finish_reason": "stop_hook_continue"},
            _later_row(kind),
        ]
        later = messages[4]["content"]
        agent._persist_user_message_idx = 4  # a stale index that already points at the later row
        agent._flush_messages_to_session_db(messages, conversation_history=[])
        assert _user_rows(agent) == ["old ask", ASK, later]
    finally:
        agent._session_db.close()


@pytest.mark.parametrize("kind", ["pre_stop", "steer"])
def test_reanchor_fallback_skips_a_nudge_or_steer_row(kind):
    """The turn row holds neither the wire text nor the override (e.g. a rewritten carrier): the
    fallback must still be the human row, not the later nudge or steer row."""
    from agent.turn_context import reanchor_current_turn_user_idx

    messages = [
        {"role": "user", "content": "old ask"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": ASK + "\n\n(rewritten)"},
        {"role": "assistant", "content": "interim answer", "finish_reason": "stop_hook_continue"},
        _later_row(kind),
    ]
    assert reanchor_current_turn_user_idx(messages, WIRE) == 2


def test_live_turn_row_after_the_turn_start_flush(tmp_path, monkeypatch):
    """Observation for the issue (no assertion on the fix): what the live row holds after a real flush."""
    agent = _agent(tmp_path, monkeypatch)
    try:
        agent._persist_user_message_override = ASK
        messages = [{"role": "user", "content": "old ask"}, {"role": "assistant", "content": "old answer"},
                    {"role": "user", "content": WIRE}]
        agent._persist_user_message_idx = 2
        agent._flush_messages_to_session_db(messages, conversation_history=[])
        print("LIVE_ROW_AFTER_FLUSH", repr(messages[2]["content"]))
        assert _user_rows(agent) == ["old ask", ASK]
    finally:
        agent._session_db.close()
