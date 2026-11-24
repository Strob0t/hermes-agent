"""Proactive client-side pacing to avoid provider 429s.

Two independent, in-process throttle dimensions per ``(provider, model)``: requests-per-minute
(``RateLimiter``) and input-tokens-per-minute (``TokenBucketLimiter``). See
``hermes_cli.rate_limits`` for the config resolvers (``providers.<id>.requests_per_minute`` /
``providers.<id>.input_tokens_per_minute``, each with a per-model override; unset/<=0 disables
that dimension).

Complements the reactive breaker in ``agent.nous_rate_guard``: this module tries to avoid the
limit in the first place. The limiters live in this process only; a multi-process gateway is
not covered.

Every limiter lock guards only O(1) bookkeeping, never a sleep: the caller sleeps after the lock
is released, so a long wait never blocks another thread's scheduling.

The configured ceilings are resolved once per agent and ``(provider, model)`` pair and cached on
the agent, so the config is not read on every API call.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from hermes_cli.rate_limits import (
    get_provider_input_tokens_per_minute,
    get_provider_requests_per_minute,
)

logger = logging.getLogger(__name__)

_NOTICE_THRESHOLD_SECONDS = 5.0
_AGENT_CACHE_ATTR = "_rate_limit_config_cache"

_clock = time.monotonic
_sleep = time.sleep


class RateLimiter:
    """Spaces calls to at most ``requests_per_minute`` per minute (single-slot scheduling, no
    burst credit banked for a caller that comes back late)."""

    def __init__(self, requests_per_minute: float) -> None:
        self._lock = threading.Lock()
        self._next_slot = 0.0
        self._interval = 60.0 / requests_per_minute
        self._recent_calls: "deque[float]" = deque()

    def update_rate(self, requests_per_minute: float) -> None:
        with self._lock:
            self._interval = 60.0 / requests_per_minute

    def wait_time(self, now: Optional[float] = None) -> float:
        """Reserve this caller's slot and return how long it should wait. Never sleeps."""
        if now is None:
            now = _clock()
        with self._lock:
            start = max(now, self._next_slot)
            self._next_slot = start + self._interval
            self._recent_calls.append(now)
        return start - now

    def current_requests(self, now: Optional[float] = None) -> int:
        """Calls to ``wait_time`` in the trailing 60 s window (display only)."""
        if now is None:
            now = _clock()
        with self._lock:
            while self._recent_calls and now - self._recent_calls[0] > 60.0:
                self._recent_calls.popleft()
            return len(self._recent_calls)


class TokenBucketLimiter:
    """Leaky bucket over confirmed input-token usage: ``capacity`` tokens drain at
    ``capacity / 60`` tokens/s. ``record_usage`` is fed provider-confirmed counts after a
    response; ``wait_time`` gates the next request and may project a rough pre-send estimate on
    top of the backlog without reserving it.

    The estimate is clamped to ``capacity``: a request larger than the whole per-minute budget
    cannot be satisfied by waiting, so an oversized estimate must not cause an unbounded delay.
    """

    def __init__(self, capacity: float) -> None:
        self._lock = threading.Lock()
        self._capacity = capacity
        self._rate = capacity / 60.0
        self._level = 0.0
        self._last_update: Optional[float] = None

    def update_capacity(self, capacity: float) -> None:
        with self._lock:
            self._capacity = capacity
            self._rate = capacity / 60.0

    def _drain_locked(self, now: float) -> None:
        if self._last_update is not None:
            elapsed = now - self._last_update
            if elapsed > 0:
                self._level = max(0.0, self._level - elapsed * self._rate)
        self._last_update = now

    def record_usage(self, tokens: float, now: Optional[float] = None) -> None:
        if now is None:
            now = _clock()
        with self._lock:
            self._drain_locked(now)
            self._level += max(0.0, tokens or 0)

    def wait_time(self, estimated_tokens: float = 0, now: Optional[float] = None) -> float:
        """Never sleeps; the lock is held only for this bookkeeping."""
        if now is None:
            now = _clock()
        with self._lock:
            self._drain_locked(now)
            effective_estimate = min(max(0.0, estimated_tokens), self._capacity)
            projected = self._level + effective_estimate
            if projected <= self._capacity:
                return 0.0
            return (projected - self._capacity) / self._rate

    def current_level(self, now: Optional[float] = None) -> float:
        """Current bucket fill after draining (display only)."""
        if now is None:
            now = _clock()
        with self._lock:
            self._drain_locked(now)
            return self._level


_limiters: Dict[Tuple[str, str], RateLimiter] = {}
_registry_lock = threading.Lock()

_token_limiters: Dict[Tuple[str, str], TokenBucketLimiter] = {}
_token_registry_lock = threading.Lock()


def _get_limiter(provider: str, model: str, requests_per_minute: float) -> RateLimiter:
    key = (provider, model)
    with _registry_lock:
        limiter = _limiters.get(key)
        if limiter is None:
            limiter = _limiters[key] = RateLimiter(requests_per_minute)
        else:
            limiter.update_rate(requests_per_minute)
        return limiter


def _get_token_limiter(provider: str, model: str, capacity: float) -> TokenBucketLimiter:
    key = (provider, model)
    with _token_registry_lock:
        limiter = _token_limiters.get(key)
        if limiter is None:
            limiter = _token_limiters[key] = TokenBucketLimiter(capacity)
        else:
            limiter.update_capacity(capacity)
        return limiter


def _resolve_limits(agent: Any, provider: str, model: str) -> Tuple[Optional[float], Optional[float]]:
    """``(requests_per_minute, input_tokens_per_minute)`` for the pair; cached on ``agent``
    so a provider/model switch re-resolves but a steady agent never re-reads the config."""
    key = (provider, model)
    cache = getattr(agent, _AGENT_CACHE_ATTR, None) if agent is not None else None
    if isinstance(cache, dict) and key in cache:
        return cache[key]
    limits = (get_provider_requests_per_minute(provider, model),
              get_provider_input_tokens_per_minute(provider, model))
    if agent is not None:
        try:
            if not isinstance(cache, dict):
                cache = {}
                setattr(agent, _AGENT_CACHE_ATTR, cache)
            cache[key] = limits
        except Exception:
            pass
    return limits


def _sum_string_lengths(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(_sum_string_lengths(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_sum_string_lengths(v) for v in value)
    return 0


def _estimate_request_tokens(request: Any) -> int:
    """Rough size estimate: characters across every string leaf divided by ~4. Never raises."""
    if not request:
        return 0
    try:
        return _sum_string_lengths(request) // 4
    except Exception:
        return 0


@dataclass(frozen=True)
class RateLimitStatus:
    """Live usage for one (provider, model); a dimension is ``None`` when not configured."""

    rpm_used: Optional[int]
    rpm_limit: Optional[float]
    tpm_used: Optional[float]
    tpm_limit: Optional[float]


def status_snapshot(provider: str, model: str, agent: Any = None) -> Optional[RateLimitStatus]:
    """Live RPM/TPM usage for the pair, or ``None`` when neither dimension is configured."""
    requests_per_minute, input_tokens_per_minute = _resolve_limits(agent, provider, model)
    if requests_per_minute is None and input_tokens_per_minute is None:
        return None
    rpm_used: Optional[int] = None
    if requests_per_minute is not None:
        rpm_used = _get_limiter(provider, model, requests_per_minute).current_requests()
    tpm_used: Optional[float] = None
    if input_tokens_per_minute is not None:
        tpm_used = _get_token_limiter(provider, model, input_tokens_per_minute).current_level()
    return RateLimitStatus(
        rpm_used=rpm_used, rpm_limit=requests_per_minute,
        tpm_used=tpm_used, tpm_limit=input_tokens_per_minute,
    )


def record_input_tokens(provider: str, model: str, tokens: float, agent: Any = None) -> None:
    """Feed provider-confirmed input-token usage into the TPM bucket; no-op when unconfigured."""
    _, capacity = _resolve_limits(agent, provider, model)
    if capacity is None:
        return
    _get_token_limiter(provider, model, capacity).record_usage(tokens)


def _notify(agent: Any, text: str) -> None:
    if agent is None:
        return
    try:
        agent._vprint(f"{getattr(agent, 'log_prefix', '')}{text}", force=True)
    except Exception:
        pass


def throttle_before_request(agent: Any, provider: str, model: str, request: Any = None) -> float:
    """Pace this outbound request against the configured ceilings for the pair and return the
    slept delay. No-op on each unconfigured dimension (default: both)."""
    provider, model = str(provider or ""), str(model or "")
    requests_per_minute, input_tokens_per_minute = _resolve_limits(agent, provider, model)
    if requests_per_minute is None and input_tokens_per_minute is None:
        return 0.0
    delay = 0.0
    if requests_per_minute is not None:
        rpm_delay = _get_limiter(provider, model, requests_per_minute).wait_time()
        logger.debug("%s/%s: pacing to %.3g req/min -> %.3fs wait", provider, model,
                     requests_per_minute, rpm_delay)
        delay = max(delay, rpm_delay)
    if input_tokens_per_minute is not None:
        estimated = _estimate_request_tokens(request)
        if estimated > input_tokens_per_minute:
            logger.warning(
                "%s/%s: estimated input (~%d tokens) exceeds the configured input_tokens_per_minute "
                "ceiling (%g); sending anyway.", provider, model, estimated, input_tokens_per_minute)
            _notify(agent, f"⚠️ Estimated input (~{estimated:,} tokens) exceeds the configured "
                           f"input_tokens_per_minute limit ({input_tokens_per_minute:g}) for "
                           f"{provider}/{model} — sending anyway.")
        tpm_delay = _get_token_limiter(provider, model, input_tokens_per_minute).wait_time(
            estimated_tokens=estimated)
        logger.debug("%s/%s: pacing to %.3g input tok/min (est=%d) -> %.3fs wait", provider, model,
                     input_tokens_per_minute, estimated, tpm_delay)
        delay = max(delay, tpm_delay)
    if delay <= 0:
        return 0.0
    if delay > _NOTICE_THRESHOLD_SECONDS:
        _notify(agent, f"⏳ Pacing requests to {provider}/{model} ({delay:.1f}s)...")
    _sleep(delay)
    return delay
