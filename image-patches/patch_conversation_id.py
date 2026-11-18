"""Local patch of Hermes: pass the compression-stable ``conversation_id`` to the generic ``pre_stop`` and
``transform_llm_output`` plugin hooks (applied after patch_stop_hook.py).

A context compaction rotates ``agent.session_id`` (agent/conversation_compression.py) but keeps the
conversation root returned by ``agent.prompt_cache_scope.conversation_id``. Tool handlers already get
``conversation_id`` from the dispatcher (model_tools.py); the two hooks did not. A plugin that keys
per-conversation state on it (the goal loop) lost that state after a compaction, because the hooks
only saw the new ``session_id``. This patch adds the field to both hook payloads. The lookup never
raises: if it fails the field is an empty string and a plugin falls back to ``session_id``.

Each old text must occur exactly once, else the build fails. Nothing is written unless all match.
"""
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/opt/hermes"

PATCHES = [
    # 1. plugins.py: the pre_stop dispatcher accepts and forwards conversation_id.
    ("hermes_cli/plugins.py",
     '''def get_pre_stop_continue_message(
    *, session_id: str = "", platform: str = "", attempt: int = 0, final_response: str = "",
) -> Optional[str]:''',
     '''def get_pre_stop_continue_message(
    *, session_id: str = "", platform: str = "", attempt: int = 0, final_response: str = "",
    conversation_id: str = "",
) -> Optional[str]:'''),
    ("hermes_cli/plugins.py",
     '''    hooks self-throttle."""
    hook_results = invoke_hook(
        "pre_stop", session_id=session_id, platform=platform, attempt=attempt,
        final_response=final_response,
    )''',
     '''    hooks self-throttle. ``conversation_id`` is the compression-stable root, so a plugin can key
    per-conversation state that survives a compaction (which rotates ``session_id``)."""
    hook_results = invoke_hook(
        "pre_stop", session_id=session_id, platform=platform, attempt=attempt,
        final_response=final_response, conversation_id=conversation_id,
    )'''),
    # 2. turn_stop_gates.py: a safe lookup helper, and the gate passes the agent's conversation_id.
    ("agent/turn_stop_gates.py",
     '''def _pre_stop_nudge(agent, final_response, attempt: int) -> Optional[str]:''',
     '''def _conversation_id_safe(agent) -> str:
    """local patch: the compression-stable conversation root for plugin hooks; never raises, so a
    failed lineage walk only drops the field (hooks then fall back to ``session_id``)."""
    try:
        from agent.prompt_cache_scope import conversation_id
        return conversation_id(agent) or ""
    except Exception:
        logger.debug("conversation_id lookup failed", exc_info=True)
        return ""


def _pre_stop_nudge(agent, final_response, attempt: int) -> Optional[str]:'''),
    ("agent/turn_stop_gates.py",
     '''                attempt=attempt, final_response=final_response,
            )
    except Exception:
        logger.debug("pre_stop hook check failed", exc_info=True)''',
     '''                attempt=attempt, final_response=final_response,
                conversation_id=_conversation_id_safe(agent),
            )
    except Exception:
        logger.debug("pre_stop hook check failed", exc_info=True)'''),
    # 3. turn_finalizer.py: add conversation_id to the transform_llm_output payload.
    ("agent/turn_finalizer.py",
     '''                   platform=platform, turn_id=turn_id)  # turn_id: per-turn identity for the hook callback gate
''',
     '''                   platform=platform, turn_id=turn_id)  # turn_id: per-turn identity for the hook callback gate
    # local patch: the compression-stable conversation root, so a plugin keyed on it still finds its
    # state after a compaction rotated session_id; an empty string if the lookup fails.
    try:
        from agent.turn_stop_gates import _conversation_id_safe
        payload["conversation_id"] = _conversation_id_safe(agent)
    except Exception:
        payload["conversation_id"] = ""
'''),
]


# A marker that ONLY this patch introduces, so the idempotence check never mistakes pristine text
# for a patched file. The old marker `conversation_id: str = ""` also occurs in the pristine
# hermes_cli/plugins.py (an unrelated middleware_trace dispatcher), so it aborted on a clean tree
# and the patch never applied in the image build.
PATCHED_MARKERS = {
    "hermes_cli/plugins.py": "final_response=final_response, conversation_id=conversation_id,",
    "agent/turn_stop_gates.py": "_conversation_id_safe",
    "agent/turn_finalizer.py": 'payload["conversation_id"] = _conversation_id_safe(agent)',
}


def already_patched(rel: str, text: str) -> bool:
    """True when this patch's own output is present. `_conversation_id_safe` covers the two files
    that get the helper; plugins.py uses the unique forwarded-arg line. A pristine tree is never
    matched, so the patch applies exactly once and a re-run aborts cleanly."""
    if "_conversation_id_safe" in text:
        return True
    marker = PATCHED_MARKERS.get(rel)
    return bool(marker) and marker in text


def main():
    texts = {}
    for rel, old, new in PATCHES:
        if rel not in texts:
            with open(f"{ROOT}/{rel}", encoding="utf-8") as f:
                texts[rel] = f.read()
            if already_patched(rel, texts[rel]):
                sys.exit(f"patch does not apply: {rel} is already patched")
        count = texts[rel].count(old)
        if count != 1:
            sys.exit(f"patch does not apply: {rel}: old text found {count} times")
        texts[rel] = texts[rel].replace(old, new)
    for rel, text in texts.items():
        with open(f"{ROOT}/{rel}", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"patched {rel}")


if __name__ == "__main__":
    main()
