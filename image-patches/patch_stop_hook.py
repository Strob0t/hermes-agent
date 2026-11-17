"""Local patch of Hermes: a generic ``pre_stop`` plugin hook at the text-answer stop gate.

Hermes has no general "the agent stopped with a text answer, a plugin may send it back" hook for
chat work: ``pre_verify`` (agent/turn_stop_gates.py) fires only after a file edit in the turn. This
patch adds a generic gate ``_pre_stop_nudge`` with NO file-edit condition and its own bound
``agent.max_stop_continuations`` (default 20). The hook is generic: it carries no goal-loop logic.
It gets ``session_id``, ``platform``, ``attempt`` and the last answer, and returns
``{"action": "continue", "message": ...}`` (or a Claude-Code Stop ``{"decision": "block", "reason": ...}``)
to continue the turn; any other return lets the turn finish. With no plugin registered on ``pre_stop``,
``has_hook`` is False and the gate is a no-op, so Hermes behaviour does not change (a test shows this).

Each old text must occur exactly once, else the build fails.
"""
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/opt/hermes"

PATCHES = [
    # 1. turn_stop_gates.py: add the _pre_stop_nudge gate function after _pre_verify_nudge.
    ("agent/turn_stop_gates.py",
     """    except Exception:
        logger.debug("pre_verify hook check failed", exc_info=True)
    return None


def _kanban_stop_nudge(agent, messages) -> Optional[str]:
""",
     """    except Exception:
        logger.debug("pre_verify hook check failed", exc_info=True)
    return None


def _pre_stop_nudge(agent, final_response, attempt: int) -> Optional[str]:
    \"\"\"local patch: a generic ``pre_stop`` hook may keep the turn going when the model stops with a
    text answer, with NO file-edit condition (unlike ``_pre_verify_nudge``). Bounded by
    ``agent.max_stop_continuations``. Carries no goal-loop logic; a plugin owns the policy.\"\"\"
    try:
        from agent.verify_hooks import max_stop_continuations
        from hermes_cli.lifecycle import has_hook
        from hermes_cli.plugins import get_pre_stop_continue_message

        if has_hook("pre_stop") and attempt < max_stop_continuations():
            return get_pre_stop_continue_message(
                session_id=getattr(agent, "session_id", None) or "",
                platform=getattr(agent, "platform", "") or "",
                attempt=attempt, final_response=final_response,
            )
    except Exception:
        logger.debug("pre_stop hook check failed", exc_info=True)
    return None


def _kanban_stop_nudge(agent, messages) -> Optional[str]:
"""),
    # 2. turn_stop_gates.py: run the new gate after the kanban gate, before the no-continue return.
    ("agent/turn_stop_gates.py",
     """        return verdict
    return StopGateVerdict(
        continue_turn=False, final_response=final_response,
        pending_verification_response=pending_verification_response,
        pending_verification_response_previewed=pending_verification_response_previewed,
    )
""",
     """        return verdict

    # local patch: generic pre_stop gate — a plugin may continue a text-answer turn with no file edit
    _stop_attempt = getattr(agent, "_pre_stop_nudges", 0)
    _stop_nudge = _pre_stop_nudge(agent, final_response, _stop_attempt)
    if _stop_nudge:
        agent._pre_stop_nudges = _stop_attempt + 1
        final_msg["finish_reason"] = "stop_hook_continue"
        _append_interim_answer(
            agent, final_msg, messages, conversation_history, "pre_stop interim flush failed"
        )
        verdict = _continue(_stop_nudge, "_pre_stop_synthetic")
        logger.debug("pre_stop nudge issued (attempt %d)", agent._pre_stop_nudges)
        return verdict
    return StopGateVerdict(
        continue_turn=False, final_response=final_response,
        pending_verification_response=pending_verification_response,
        pending_verification_response_previewed=pending_verification_response_previewed,
    )
"""),
    # 3. verify_hooks.py: add the bound helper and its default next to max_verify_nudges.
    ("agent/verify_hooks.py",
     """def coding_verify_guidance(config: Optional[dict[str, Any]] = None) -> Optional[str]:
""",
     """def max_stop_continuations(config: Optional[dict[str, Any]] = None) -> int:
    \"\"\"local patch: bound on consecutive ``pre_stop`` continue directives per turn (>= 0).\"\"\"
    try:
        return max(0, int(_agent_cfg(config).get("max_stop_continuations")))
    except (TypeError, ValueError):
        return DEFAULT_MAX_STOP_CONTINUATIONS


def coding_verify_guidance(config: Optional[dict[str, Any]] = None) -> Optional[str]:
"""),
    ("agent/verify_hooks.py",
     """DEFAULT_MAX_VERIFY_NUDGES = 3
""",
     """DEFAULT_MAX_VERIFY_NUDGES = 3
DEFAULT_MAX_STOP_CONTINUATIONS = 20  # local patch: default bound for the generic pre_stop gate
"""),
    ("agent/verify_hooks.py",
     """__all__ = ["CODING_VERIFY_GUIDANCE", "DEFAULT_MAX_VERIFY_NUDGES", "coding_verify_guidance", "max_verify_nudges"]""",
     """__all__ = ["CODING_VERIFY_GUIDANCE", "DEFAULT_MAX_STOP_CONTINUATIONS", "DEFAULT_MAX_VERIFY_NUDGES",
           "coding_verify_guidance", "max_stop_continuations", "max_verify_nudges"]"""),
    # 4. plugins.py: register the generic pre_stop hook name.
    ("hermes_cli/plugins.py",
     """    "pre_verify", "pre_api_request", "post_api_request", "api_request_error",
""",
     """    "pre_verify",
    # local patch: pre_stop — once per turn when the model stops with a text answer, with NO
    # file-edit condition (unlike pre_verify). Return {"action": "continue", "message"} (or a
    # Claude-Code Stop {"decision": "block", "reason"}) to continue the turn; anything else finishes.
    # Bounded by agent.max_stop_continuations (default 20). Generic: carries no goal-loop logic.
    "pre_stop",
    "pre_api_request", "post_api_request", "api_request_error",
"""),
    # 5. plugins.py: add the dispatcher after get_pre_verify_continue_message.
    ("hermes_cli/plugins.py",
     """        if action in (\"continue\", \"block\") and isinstance(message, str) and message.strip():
            return message.strip()
    return None


def get_plugin_error_classification(
""",
     """        if action in (\"continue\", \"block\") and isinstance(message, str) and message.strip():
            return message.strip()
    return None


def get_pre_stop_continue_message(
    *, session_id: str = \"\", platform: str = \"\", attempt: int = 0, final_response: str = \"\",
) -> Optional[str]:
    \"\"\"local patch: check generic ``pre_stop`` hooks for ``{\"action\": \"continue\", \"message\"}`` (or a
    Claude-Code Stop ``{\"decision\": \"block\", \"reason\"}``) to keep a text-answer turn going with no
    file edit; first non-empty message wins, any other return lets the turn finish. ``attempt`` lets
    hooks self-throttle.\"\"\"
    hook_results = invoke_hook(
        \"pre_stop\", session_id=session_id, platform=platform, attempt=attempt,
        final_response=final_response,
    )
    for result in hook_results:
        if not isinstance(result, dict):
            continue
        action = str(result.get(\"action\") or result.get(\"decision\") or \"\").strip().lower()
        message = result.get(\"message\") or result.get(\"reason\")
        if action in (\"continue\", \"block\") and isinstance(message, str) and message.strip():
            return message.strip()
    return None


def get_plugin_error_classification(
"""),
]


def main():
    texts = {}
    for rel, old, new in PATCHES:
        if rel not in texts:
            with open(f"{ROOT}/{rel}", encoding="utf-8") as f:
                texts[rel] = f.read()
            if "local patch:" in texts[rel] and "pre_stop" in texts[rel]:
                sys.exit(f"patch does not apply: {rel} is already patched")
        count = texts[rel].count(old)
        if count != 1:
            sys.exit(f"patch does not apply: {rel}: old text found {count} times")
        texts[rel] = texts[rel].replace(old, new)
    # nothing is written unless every replacement matched
    for rel, text in texts.items():
        with open(f"{ROOT}/{rel}", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"patched {rel}")


if __name__ == "__main__":
    main()
