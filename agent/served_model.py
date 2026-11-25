"""Per-response *served model* capture for routing proxies (#54864).

A LiteLLM-style proxy answers with the configured alias in the body's ``model`` field and puts
the deployment it really routed to in a response header (``x-litellm-model-id``, else the
upstream ``x-litellm-model-api-base``). The OpenAI SDK's parsed objects drop headers, so the
capture rides an ``httpx`` response hook on the client Hermes builds for the agent; it stores
the header onto ``agent.last_served_model`` (``None`` when the response carried none, so a
value never outlives the request that produced it). Consumers: ``agent/turn_finalizer.py``
(result ``served_model`` / ``requested_model``) and the opt-in gateway footer field
``served_model`` (``gateway/runtime_footer.py``). Fail-open everywhere.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

SERVED_MODEL_HEADERS: tuple[str, ...] = ("x-litellm-model-id", "x-litellm-model-api-base")
# ctx-window patch: Bifrost names the chain model that served an alias in these two answer headers
BIFROST_PROVIDER_HEADER = "x-bifrost-provider"
BIFROST_MODEL_HEADER = "x-bifrost-resolved-model"
_WINDOW_SERVED_ATTR = "_ctx_window_served_model"
# the served model and its window for each session: an agent is built for each turn, the checks before a turn
# read this record. Kept in the process only; a record older than the TTL is not used.
SESSION_WINDOW_TTL = 3600.0
SESSION_WINDOW_MAX = 2000
_session_windows: dict[str, tuple[str, int, float]] = {}
_HOOK_MARK = "_hermes_served_model_hook"


def served_model_from_headers(headers: Any) -> Optional[str]:
    """First non-empty served-model header, or ``None``."""
    if headers is None or not hasattr(headers, "get"):
        return None
    for name in SERVED_MODEL_HEADERS:
        value = headers.get(name)
        if value not in (None, ""):
            return str(value).strip()
    model = str(headers.get(BIFROST_MODEL_HEADER) or "").strip()
    if model:
        provider = str(headers.get(BIFROST_PROVIDER_HEADER) or "").strip()
        return f"{provider}/{model}" if provider and not model.startswith(provider + "/") else model
    return None


def served_model_window(agent: Any, served: Optional[str]) -> Optional[int]:
    """The window the model list of the agent's endpoint gives for the served model (exact id), else None."""
    base_url = str(getattr(agent, "base_url", "") or "")
    if not served or not base_url:
        return None
    from agent.model_metadata import fetch_endpoint_model_metadata
    entry = fetch_endpoint_model_metadata(base_url, api_key=getattr(agent, "api_key", "") or "").get(served)
    window = entry.get("context_length") if isinstance(entry, dict) else None
    return window if isinstance(window, int) and not isinstance(window, bool) and window > 0 else None


def _session_keys(agent: Any) -> list[str]:
    """The keys a session record is stored under: the gateway session key and the session id of the agent."""
    keys = (getattr(agent, "_gateway_session_key", None), getattr(agent, "session_id", None))
    return [str(key) for key in keys if key]


def remember_session_window(agent: Any, served: str, window: int) -> None:
    """Store the served model and its window for the session of the agent."""
    import time
    keys = _session_keys(agent)
    if not keys:
        return
    if len(_session_windows) >= SESSION_WINDOW_MAX:
        oldest = sorted(list(_session_windows.items()), key=lambda item: item[1][2])[:SESSION_WINDOW_MAX // 10]
        for key, _ in oldest:
            _session_windows.pop(key, None)
    for key in keys:
        _session_windows[key] = (served, window, time.time())


def session_window(*keys: Any) -> Optional[tuple[str, int]]:
    """(served model, window) of the newest record of these session keys that is not too old, else None."""
    import time
    best = None
    for key in keys:
        entry = _session_windows.get(str(key)) if key else None
        if entry and time.time() - entry[2] <= SESSION_WINDOW_TTL and (best is None or entry[2] > best[2]):
            best = entry
    return (best[0], best[1]) if best else None


def session_window_or(default: int, *keys: Any) -> int:
    """The window of the model that served the session last, else ``default`` (the window of the alias row)."""
    try:
        found = session_window(*keys)
    except Exception:
        return default
    if not found or found[1] == default:
        return default
    logger.info("Session hygiene uses the window of the served model %s: %s tokens (alias window %s)",
                found[0], found[1], default)
    return found[1]


def apply_session_window(agent: Any) -> None:
    """Before a turn: set the compressor window to the window of the model that served the session last."""
    try:
        compressor = getattr(agent, "context_compressor", None)
        found = session_window(*_session_keys(agent))
        if compressor is None or not found or not hasattr(compressor, "context_length"):
            return
        previous = compressor.context_length
        if previous == found[1]:
            return
        compressor.context_length = found[1]
        logger.info("Context window starts from the model that served the session last, %s: %s -> %s tokens "
                    "(trigger %s)", found[0], previous, found[1], getattr(compressor, "threshold_tokens", None))
    except Exception:
        logger.debug("session window skipped", exc_info=True)


def apply_served_window(agent: Any) -> None:
    """Set the compressor window to the window of the served model, once per change of the served model."""
    served = getattr(agent, "last_served_model", None)
    if not served or served == getattr(agent, _WINDOW_SERVED_ATTR, None):
        return
    setattr(agent, _WINDOW_SERVED_ATTR, served)
    window = served_model_window(agent, served)
    if window is None:
        return
    remember_session_window(agent, served, window)
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None or getattr(compressor, "context_length", None) == window:
        return
    previous = getattr(compressor, "context_length", None)
    compressor.context_length = window
    logger.info("Context window follows the served model %s: %s -> %s tokens (trigger %s)",
                served, previous, window, getattr(compressor, "threshold_tokens", None))


def install_served_model_capture(agent: Any, client: Any) -> None:
    """Register the response hook on *client*'s ``httpx`` transport (idempotent per client)."""
    http_client = getattr(client, "_client", None)
    hooks = getattr(http_client, "event_hooks", None)
    if not isinstance(hooks, dict):
        return
    if any(getattr(h, _HOOK_MARK, False) for h in hooks.get("response", ())):
        return

    def _on_response(response: Any) -> None:
        try:
            if 200 <= int(getattr(response, "status_code", 0)) < 300:
                agent.last_served_model = served_model_from_headers(getattr(response, "headers", None))
                apply_served_window(agent)
        except Exception:
            logger.debug("served-model header capture skipped", exc_info=True)

    setattr(_on_response, _HOOK_MARK, True)
    try:
        # httpx copies on assignment; rebuild the mapping instead of mutating the live list.
        http_client.event_hooks = {**hooks, "response": [*hooks.get("response", ()), _on_response]}
    except Exception:
        logger.debug("served-model hook install skipped", exc_info=True)


def result_model_fields(agent: Any) -> dict[str, Optional[str]]:
    """``requested_model`` / ``served_model`` for the turn result: the proxy header when the
    last response carried one, else Hermes' own fallback route (primary → active model)."""
    served = getattr(agent, "last_served_model", None)
    requested = agent.model
    if not served and getattr(agent, "_fallback_activated", False):
        primary = str((getattr(agent, "_primary_runtime", None) or {}).get("model") or "").strip()
        if primary and primary != agent.model:
            requested, served = primary, agent.model
    return {"requested_model": requested, "served_model": served or None}
