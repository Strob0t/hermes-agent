"""JSON whitespace compaction at the tool-result append boundary.

Only insignificant whitespace is removed: string literals, number spellings,
key order and even duplicate keys survive byte-for-byte. No parse/reserialize
round trip, auxiliary model, recovery call or rewrite of past messages is needed.
The ``tool_output.json_compaction`` policy is read once per process and cached.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
import json
import logging
import re
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Bound work on untrusted results (including tools exempt from spillover).
_MAX_INPUT_CHARS = 1_000_000
_MIN_CHARS_FLOOR = 1024
_JSON_STRING_OR_SPACE = re.compile(r'"(?:[^"\\]|\\.)*"|[ \t\r\n]+')

_policy_lock = threading.Lock()
_policy_cache: Optional[tuple] = None
_DISABLED = ("off", (), 0, 0.0)


def _reject_constant(value: str):
    raise ValueError(f"Non-JSON numeric constant: {value}")


def _compact_json(content: str) -> str:
    """Validate first, then remove only whitespace outside JSON string literals."""
    if not content.lstrip(" \t\r\n").startswith(("{", "[")):
        return content
    try:
        # Validation must not round floats, normalize escapes or hit Python's
        # integer-digit conversion limit. The parsed object is never serialized.
        json.loads(content, parse_int=lambda _: None, parse_float=lambda _: None,
                   parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return content
    return _JSON_STRING_OR_SPACE.sub(
        lambda match: match[0] if match[0].startswith('"') else "", content,
    )


def _read_policy() -> tuple:
    """Return ``(mode, exclude_tools, min_chars, min_savings_ratio)``; invalid = disabled."""
    try:
        from hermes_cli.config import load_config_readonly
        policy = load_config_readonly().get("tool_output", {}).get("json_compaction", {})
    except (AttributeError, TypeError, ValueError, OSError, ImportError):
        return _DISABLED
    if not isinstance(policy, dict):
        return _DISABLED
    mode = policy.get("mode", "off")
    if mode not in ("observe", "compact"):
        return _DISABLED
    excluded = policy.get("exclude_tools", ["read_file"])
    if not isinstance(excluded, list) or any(not isinstance(p, str) for p in excluded):
        return _DISABLED
    minimum = policy.get("min_chars", _MIN_CHARS_FLOOR)
    ratio = policy.get("min_savings_ratio", 0.1)
    if (type(minimum) is not int or minimum < _MIN_CHARS_FLOOR
            or type(ratio) not in (int, float) or not 0 < ratio < 1):
        return _DISABLED
    return mode, tuple(excluded), minimum, float(ratio)


def _get_policy() -> tuple:
    global _policy_cache
    policy = _policy_cache
    if policy is None:
        with _policy_lock:
            if _policy_cache is None:
                _policy_cache = _read_policy()
            policy = _policy_cache
    return policy


def reset_policy_cache() -> None:
    """Drop the cached policy so the next call reads the config again."""
    global _policy_cache
    with _policy_lock:
        _policy_cache = None


def compact_tool_result(content, tool_name: str):
    """Apply the configured policy to a new text result; off/malformed = unchanged.

    Called after per-result spillover, before hints/wrapping/append. Spill files
    and UI results retain their original formatting. Multimodal results bypass it.
    ``observe`` measures characters only and never replaces the result; token and
    monetary savings depend on the model tokenizer and subsequent use.
    """
    if not isinstance(content, str) or not _MIN_CHARS_FLOOR <= len(content) <= _MAX_INPUT_CHARS:
        return content
    mode, excluded, minimum, ratio = _get_policy()
    if mode not in ("observe", "compact") or len(content) < minimum:
        return content
    if any(fnmatchcase(tool_name, pattern) for pattern in excluded):
        return content
    compacted = _compact_json(content)
    if len(compacted) > len(content) * (1 - ratio):
        return content
    logger.info(
        "JSON tool-result compaction: mode=%s tool=%s original_chars=%d compact_chars=%d",
        mode, tool_name, len(content), len(compacted),
    )
    return compacted if mode == "compact" else content
