"""Post-tool-round false-stop detection (opt-in, ``agent.false_stop_detection``).

A model sometimes ends a turn with ``finish_reason=stop`` on text that announces the next
action ("Next I will run the tests:", "接下来看日志。") right after a tool round, so the
tool call it meant to make is lost. This module only decides; the turn loop adds the
nudge as a synthetic user row and continues. Bounded to ``MAX_FALSE_STOP_NUDGES`` per
tool round. The turn-start sibling is the intent-ack continuation.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

MAX_FALSE_STOP_NUDGES = 2
FALSE_STOP_NUDGE = (
    "Your previous response ended as if you were about to call a tool, but no tool call "
    "was made. Continue the task by issuing the actual tool call now. Do not narrate "
    "intent - make the call directly."
)

_CJK_INTENT_MARKERS = (
    "继续", "接下来", "下一步", "然后", "先读", "先看", "先把", "接着看", "再看", "继续看",
    "再读", "随后", "接着", "马上", "即将", "先快速", "先确认", "先检查", "再实现", "开始",
    "先梳理",
)
# Word-bounded English markers. Bare "now" / "next" are left out: a finished answer
# often uses them ("is now fixed", "next release").
_EN_INTENT_RE = re.compile(
    r"\b(?:let me|i'll|i will|i am going to|i'm going to|next,? i|then i|continue|proceed"
    r"|proceeding to|moving on to|carry on|read the rest)\b",
    re.IGNORECASE,
)
_DONE_MARKERS = (
    "已完成", "全部完成", "已搞定", "结论如下", "总结如下", "如上", "以上是",
    "done", "finished", "completed",
)
_WAIT_MARKERS = (
    "等你", "等待", "仍在跑", "正在跑", "仍在运行", "等确认", "等决定", "还在跑", "还在运行",
    "waiting for", "still running", "pending",
)
_QUESTION_MARKERS = (
    "你决定", "你来决定", "由你决定", "取决于你", "建议你", "你可以", "请确认", "你确认",
    "你看", "如何？", "怎么样？", "要不要", "是否需要", "还是",
)
# ASCII "." is not an ending here: English answers end in "." and would over-trigger.
_CONTINUATION_ENDINGS = ("。", "！", ";", "；", "…")
_MAX_PREAMBLE_CHARS = 120
_MAX_NARRATION_CHARS = 200

_ON = {"1", "true", "yes", "on"}
_OFF = {"0", "false", "no", "off"}


def _session_is_messaging_surface() -> bool:
    try:
        from gateway.session_context import session_is_messaging_surface
        return session_is_messaging_surface()
    except Exception:
        return False


def false_stop_detection_enabled(config: Optional[dict[str, Any]] = None) -> bool:
    """``agent.false_stop_detection``: ``off`` (default) | ``on`` | ``auto`` (on except on
    messaging surfaces). A bool is accepted; a missing or unknown value is off."""
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly()
        except Exception as exc:
            logger.debug("agent.false_stop_detection unreadable; detection off: %s", exc)
            return False
    agent_cfg = config.get("agent") if isinstance(config, dict) else None
    value = agent_cfg.get("false_stop_detection") if isinstance(agent_cfg, dict) else None
    if isinstance(value, bool):
        return value
    token = value.strip().lower() if isinstance(value, str) else ""
    if token in _ON:
        return True
    if token == "auto":
        return not _session_is_messaging_surface()
    return False


def _tool_round_this_turn(messages: Any) -> bool:
    """An assistant tool-call row after the turn's real user row (own nudge rows skipped)."""
    for msg in reversed(messages or ()):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user" and msg.get("content") != FALSE_STOP_NUDGE:
            return False
        if role == "assistant" and msg.get("tool_calls"):
            return True
    return False


def _announces_continuation(text: str) -> bool:
    lowered = text.lower()
    return (
        (any(m in text for m in _CJK_INTENT_MARKERS) or bool(_EN_INTENT_RE.search(text)))
        and not any(m in lowered for m in _DONE_MARKERS)
        and not any(m in lowered for m in _WAIT_MARKERS)
        and not any(m in text for m in _QUESTION_MARKERS)
    )


def build_false_stop_nudge(content: str, messages: Any, attempts: int) -> Optional[str]:
    """Return the nudge text when ``content`` is a false stop after a tool round, else None.

    A: short (<120 chars) text that ends on ":" / "：" - a tool-call preamble without the call.
    B: text (<=200 chars) that ends on CJK sentence punctuation, ";" or "…", has an intent
    marker and no done / wait / question marker.
    """
    stripped = (content or "").strip()
    if not stripped or attempts >= MAX_FALSE_STOP_NUDGES or not _tool_round_this_turn(messages):
        return None
    preamble = len(stripped) < _MAX_PREAMBLE_CHARS and stripped[-1] in (":", "：")
    narration = (
        len(stripped) <= _MAX_NARRATION_CHARS
        and stripped.endswith(_CONTINUATION_ENDINGS)
        and _announces_continuation(stripped)
    )
    return FALSE_STOP_NUDGE if preamble or narration else None


__all__ = [
    "FALSE_STOP_NUDGE", "MAX_FALSE_STOP_NUDGES",
    "build_false_stop_nudge", "false_stop_detection_enabled",
]
