"""Local patch of Hermes v0.23.0: compaction anchors on the live API turn and never restates an answered ask.

session_ingress marks every API turn internal, so its user row is stored as internal_notification and the
compressor treats the live request as machinery: the summary demotes it and an older, already answered request
is restated as STILL IN PROGRESS. The patch tags a client API turn (metadata api_inbound), lets the compressor
anchor on it, and makes the in-flight scan stop at any final answer after the ask.
Each old text must occur exactly once, else the build fails.
"""
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/opt/hermes"

PATCHES = [
    ("gateway/session_ingress.py",
     """                event.internal = True  # trust comes from the private binding and preclaim, never client JSON
""",
     """                if not event.internal:
                    # local patch: a client turn, not a self-injected one; compaction anchors on it
                    event.metadata['api_inbound'] = True
                event.internal = True  # trust comes from the private binding and preclaim, never client JSON
"""),
    ("gateway/response_filters.py",
     """    return {**display_metadata_for_event(event), **admission_submission(event)}
""",
     """    api_inbound = {"api_inbound": True} if (getattr(event, "metadata", None) or {}).get("api_inbound") else {}
    return {**display_metadata_for_event(event), **admission_submission(event), **api_inbound}
"""),
    ("agent/context_compressor.py",
     """        display_kind = message.get("display_kind")
        if (display_kind and display_kind != STEER_DISPLAY_KIND) or cls._is_context_summary_message(message):
            return False
        return not cls._is_blank_user_turn(message)
""",
     """        display_kind = message.get("display_kind")
        if (display_kind and display_kind != STEER_DISPLAY_KIND and not cls._is_api_inbound_user_turn(message)) \\
                or cls._is_context_summary_message(message):
            return False
        return not cls._is_blank_user_turn(message)

    @staticmethod
    def _is_api_inbound_user_turn(message: Any) -> bool:
        \"\"\"local patch: an API client turn is stored as internal_notification, but it is human input.\"\"\"
        metadata = message.get("display_metadata") if isinstance(message, dict) else None
        return (message.get("display_kind") == "internal_notification" and isinstance(metadata, dict)
                and metadata.get("api_inbound") is True)
"""),
    ("agent/context_compressor.py",
     """            if msg.get("role") != "user" or cls._is_synthetic_compression_user_turn(msg) or msg.get("display_kind"):
""",
     """            if msg.get("role") != "user" or cls._is_synthetic_compression_user_turn(msg) or (
                    msg.get("display_kind") and not cls._is_api_inbound_user_turn(msg)):
"""),
    ("agent/context_compressor.py",
     """        for msg in reversed(messages[last_user_idx + 1:]):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                # Trailing tool result (or anything else): still mid-task.
                break
            if msg.get("tool_calls"):
                break
            if _content_text_for_contains(msg.get("content")).strip():
""",
     """        # local patch: a final answer anywhere after the ask finishes it, also when a notice or its tool
        # round follows (upstream scanned only the transcript end); a provisional verify answer does not
        for msg in messages[last_user_idx + 1:]:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            if msg.get("tool_calls") or msg.get("finish_reason") in {"verification_required", "verify_hook_continue"}:
                continue
            if _content_text_for_contains(msg.get("content")).strip():
"""),
]


def main():
    texts = {}
    for rel, old, new in PATCHES:
        if rel not in texts:
            with open(f"{ROOT}/{rel}", encoding="utf-8") as f:
                texts[rel] = f.read()
            if "local patch:" in texts[rel]:
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
