"""Opt-in running session summary: ``<HERMES_HOME>/sessions/<session_id>/running_summary.md``.

Actions: ``write`` (overwrite), ``append`` (adds a ``## HH:MM`` entry), ``read`` (empty string when absent).
The tool lives in its own ``session_summary`` toolset, outside the core and every platform bundle,
so only a session that enables that toolset sees it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

SUMMARY_FILENAME = "running_summary.md"
MAX_SUMMARY_BYTES = 64 * 1024


def _result(**payload) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _error(message: str) -> str:
    return _result(success=False, error=message)


def _valid_session_id(session_id: str) -> bool:
    return (
        bool(session_id) and session_id not in {".", ".."} and ".." not in session_id
        and "/" not in session_id and "\\" not in session_id and "\x00" not in session_id
    )


def _current_session_id() -> str:
    try:
        from gateway.session_context import get_session_env
        return get_session_env("HERMES_SESSION_ID", "") or ""
    except Exception:
        return ""


def _resolve_summary_path(session_id: str) -> Path:
    return get_hermes_home() / "sessions" / session_id / SUMMARY_FILENAME


def session_summary(
    action: str,
    session_id: Optional[str] = None,
    content: Optional[str] = None,
    current_session_id: Optional[str] = None,
) -> str:
    """Run one action on the running summary. The current session id wins over *session_id*."""
    sid = current_session_id or _current_session_id() or session_id or ""
    if not sid:
        return _error("No session id: none is bound to this session and none was given.")
    if not _valid_session_id(sid):
        return _error(f"Invalid session_id {sid!r}: path separators and '..' are not allowed.")
    path = _resolve_summary_path(sid)

    if action == "read":
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        return _result(success=True, content=text)

    if action not in {"write", "append"}:
        return _error(f"Unknown action: {action!r}. Valid actions: write, append, read.")
    if content is None:
        return _error(f"content is required for {action} action.")

    existing = path.read_text(encoding="utf-8") if action == "append" and path.is_file() else ""
    if action == "append":
        entry = f"## {datetime.now().strftime('%H:%M')}\n{content}\n"
        new_text = f"{existing.rstrip()}\n\n{entry}" if existing.strip() else entry
    else:
        new_text = content
    size = len(new_text.encode("utf-8"))
    if size > MAX_SUMMARY_BYTES:
        return _error(
            f"Refused: the summary would be {size} bytes, above the {MAX_SUMMARY_BYTES}-byte limit. "
            "Use write with a shorter summary.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text, encoding="utf-8")
    return _result(success=True, path=str(path), bytes=size)


SESSION_SUMMARY_SCHEMA = {
    "name": "session_summary",
    "description": (
        "Keep a running summary of the current session. Record decisions, errors, changed files and "
        "durable facts; the file survives context compression, so read it back when you need context "
        "beyond the recent turns.\n\n"
        "ACTIONS:\n"
        "- write: overwrite the summary (full refresh).\n"
        "- append: add a timestamped '## HH:MM' entry (incremental update).\n"
        "- read: return the current summary.\n\n"
        f"The file is limited to {MAX_SUMMARY_BYTES // 1024} KB; when append is refused, write a "
        "shorter summary. Keep it compact: no transcripts or routine tool output."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["write", "append", "read"],
                "description": "write (overwrite), append (timestamped entry), read (current summary).",
            },
            "content": {
                "type": "string",
                "description": "Text to write or append. Required for write and append.",
            },
            "session_id": {
                "type": "string",
                "description": "Only when no session is bound; the current session is used otherwise.",
            },
        },
        "required": ["action"],
    },
}


from tools.registry import registry  # noqa: E402

registry.register(
    name="session_summary",
    toolset="session_summary",
    schema=SESSION_SUMMARY_SCHEMA,
    handler=lambda args, **kw: session_summary(
        action=args.get("action", ""),
        session_id=args.get("session_id"),
        content=args.get("content"),
        current_session_id=kw.get("session_id"),
    ),
    emoji="📝",
)
