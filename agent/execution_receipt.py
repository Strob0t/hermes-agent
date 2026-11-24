"""Opt-in per-turn execution receipts (``agent.execution_receipts.enabled``, default off).

A receipt records the facts Hermes observed at its own terminal execution seams for one turn:
each main-loop provider request that reached the provider callback, each top-level tool dispatch,
and the decision of every execution middleware frame (opaque registration IDs only). The turn
decorator in ``agent.conversation_loop`` finalizes it on every exit path (normal end, exception,
interrupt). The file is owner-only (0o600), published atomically and never replaces an existing
receipt. When the feature is off no receipt is active, and each hook is one ContextVar read.
"""

from __future__ import annotations

import contextvars
import copy
import hashlib
import json
import logging
import os
import stat
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

RECEIPT_SCHEMA_VERSION = "hermes.turn-execution-receipt/v1"

_MIDDLEWARE_ID_PREFIX = "mw-sha256:"
_MIDDLEWARE_DECISIONS = frozenset({"continued", "short_circuited", "failed_open", "raised"})
_INEXACT_REASONS = frozenset({
    "codex_app_server_runtime_unobserved",
    "middleware_provenance_unattributed",
    "provider_execution_runner_mismatch",
    "request_identity_mismatch",
    "tool_execution_runner_mismatch",
})

_active_receipt: contextvars.ContextVar[Optional["TurnExecutionReceipt"]] = contextvars.ContextVar(
    "hermes_turn_execution_receipt", default=None)
_active_tool_dispatch: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "hermes_turn_execution_receipt_tool_dispatch", default=False)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identity_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def opaque_tool_call_id(value: str) -> str:
    """Receipt-safe stable identity for a provider-controlled tool call ID."""
    material = b"hermes.tool-call-id/v1\x00" + str(value).encode("utf-8", errors="surrogatepass")
    return f"tc-sha256:{hashlib.sha256(material).hexdigest()}"


def execution_receipts_enabled(agent_config: Any) -> bool:
    """``agent.execution_receipts.enabled`` from an ``agent`` config section; only ``True`` enables."""
    section = agent_config.get("execution_receipts") if isinstance(agent_config, dict) else None
    return isinstance(section, dict) and section.get("enabled") is True


def active_turn_receipt() -> Optional["TurnExecutionReceipt"]:
    return _active_receipt.get()


def get_turn_execution_receipt_dir(session_id: str) -> Path:
    return get_hermes_home() / "runtime" / "turn-execution-receipts" / _identity_key(session_id or "session")


def get_turn_execution_receipt_path(session_id: str, turn_id: str) -> Path:
    return get_turn_execution_receipt_dir(session_id) / f"{_identity_key(turn_id)}.json"


def _process_identity() -> tuple[int, str]:
    try:
        from gateway.status import get_gateway_execution_receipt_identity
        return get_gateway_execution_receipt_identity()
    except Exception:
        pid = os.getpid()
        return pid, f"process:{pid}:{uuid.uuid4().hex}"


# --- atomic, no-replace publication -------------------------------------------------------------

def _directory_open_flags() -> int:
    return (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0))


def _validate_owned_directory(fd: int, *, label: str) -> None:
    metadata = os.fstat(fd)
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError(f"{label} is not a directory")
    geteuid = getattr(os, "geteuid", None)
    if callable(geteuid) and metadata.st_uid != geteuid():
        raise PermissionError(f"{label} is not owned by the current user")
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise PermissionError(f"{label} is writable by another user")


def _open_directory_at(parent_fd: int, name: str, *, label: str, owner_only: bool) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    fd = os.open(name, _directory_open_flags(), dir_fd=parent_fd)
    try:
        _validate_owned_directory(fd, label=label)
        if owner_only:
            os.fchmod(fd, 0o700)
        os.fsync(fd)
        os.fsync(parent_fd)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _write_all(fd: int, encoded: bytes) -> None:
    view = memoryview(encoded)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("turn execution receipt write made no progress")
        view = view[written:]


def _write_receipt_posix(path: Path, encoded: bytes) -> None:
    """Stage an O_EXCL 0o600 temp file, then ``link()`` it to the final name: ``link`` fails with
    FileExistsError instead of replacing a receipt that is already there."""
    hermes_home = get_hermes_home()
    hermes_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    home_fd = os.open(hermes_home, _directory_open_flags())
    runtime_fd = root_fd = session_fd = None
    temporary_name: Optional[str] = None
    try:
        _validate_owned_directory(home_fd, label="Hermes home")
        runtime_fd = _open_directory_at(home_fd, "runtime", label="Hermes runtime directory", owner_only=False)
        root_fd = _open_directory_at(runtime_fd, "turn-execution-receipts",
                                     label="turn execution receipt root", owner_only=True)
        session_fd = _open_directory_at(root_fd, path.parent.name,
                                        label="turn execution receipt session directory", owner_only=True)
        temporary_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(temporary_name, flags, 0o600, dir_fd=session_fd)
        try:
            os.fchmod(fd, 0o600)
            _write_all(fd, encoded)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.link(temporary_name, path.name, src_dir_fd=session_fd, dst_dir_fd=session_fd, follow_symlinks=False)
        os.unlink(temporary_name, dir_fd=session_fd)
        temporary_name = None
        os.fsync(session_fd)
    finally:
        if session_fd is not None and temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=session_fd)
            except FileNotFoundError:
                pass
        for fd in (session_fd, root_fd, runtime_fd, home_fd):
            if fd is not None:
                os.close(fd)


def _write_receipt_windows(path: Path, encoded: bytes) -> None:
    """Platforms without ``dir_fd``: ``os.rename`` is atomic and refuses an existing target on Windows."""
    for directory in (get_hermes_home(), get_hermes_home() / "runtime", path.parent.parent, path.parent):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if directory.is_symlink():
            raise OSError(f"refusing receipt directory link: {directory}")
    if path.exists():
        raise FileExistsError(str(path))
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        _write_all(fd, encoded)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.rename(temporary, path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        temporary.unlink(missing_ok=True)
        raise


def _write_receipt(path: Path, encoded: bytes) -> None:
    if os.name == "nt":
        _write_receipt_windows(path, encoded)
    else:
        _write_receipt_posix(path, encoded)


# --- in-memory aggregate -------------------------------------------------------------------------

def _sanitize_frames(provenance: Any) -> tuple[list[dict[str, str]], bool]:
    """Keep only host-assigned opaque middleware IDs and fixed decision codes."""
    if provenance is None:
        return [], True
    exact = True
    frames: list[dict[str, str]] = []
    for entry in getattr(provenance, "frames", ()) or ():
        middleware_id = entry.get("middleware_id") if isinstance(entry, dict) else None
        decision = entry.get("decision") if isinstance(entry, dict) else None
        digest = middleware_id[len(_MIDDLEWARE_ID_PREFIX):] if isinstance(middleware_id, str) else ""
        if (isinstance(middleware_id, str) and middleware_id.startswith(_MIDDLEWARE_ID_PREFIX)
                and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
                and decision in _MIDDLEWARE_DECISIONS):
            frames.append({"middleware_id": middleware_id, "decision": decision})
        else:
            exact = False
    return frames, exact


class _RequestProbe:
    """One execution-middleware run; its terminal callback may run zero or one time."""

    def __init__(self, receipt: "TurnExecutionReceipt", kind: str, identity: dict[str, Any]) -> None:
        self._receipt = receipt
        self._kind = kind
        self._identity = identity
        self._event: Optional[dict[str, Any]] = None
        self._closed = False

    def dispatch(self, execute: Callable[[], Any]) -> Any:
        if self._event is not None:
            raise RuntimeError(f"{self._kind} terminal dispatch is single-use")
        self._event = self._receipt._begin_execution(self._kind, self._identity)
        token = _active_tool_dispatch.set(True) if self._kind == "tool" else None
        try:
            result = execute()
        except BaseException:
            self._receipt._finish_execution(self._event, "raised")
            raise
        finally:
            if token is not None:
                _active_tool_dispatch.reset(token)
        self._receipt._finish_execution(self._event, "returned")
        return result

    def close(self, provenance: Any = None) -> None:
        if self._closed:
            return
        self._closed = True
        self._receipt._record_decision(self._kind, self._identity, self._event, provenance)


class TurnExecutionReceipt:
    """Thread-safe aggregate for one turn, persisted once by :meth:`finalize`."""

    def __init__(self, session_id: str, *, gateway_pid: int, gateway_runtime_id: str) -> None:
        self.session_id = session_id
        self.gateway_pid = gateway_pid
        self.gateway_runtime_id = gateway_runtime_id
        self.receipt_id = uuid.uuid4().hex
        self.started_at = _utc_now()
        self.provider_executions: list[dict[str, Any]] = []
        self.tool_executions: list[dict[str, Any]] = []
        self.request_decisions: list[dict[str, Any]] = []
        self.tool_decisions: list[dict[str, Any]] = []
        self._inexact: set[str] = set()
        self._attempts: dict[str, int] = {}
        self._lock = threading.Lock()
        self._sealed = False

    def begin_provider_request(self, api_request_id: str, *, request_middleware_applied: int = 0) -> _RequestProbe:
        return _RequestProbe(self, "provider", {
            "api_request_id": str(api_request_id or ""),
            "request_middleware_applied": int(request_middleware_applied or 0),
        })

    def begin_tool_request(self, *, tool_name: str, tool_call_id: str, api_request_id: str) -> Optional[_RequestProbe]:
        """None inside an already-recorded tool dispatch: a nested runner is the same execution."""
        if _active_tool_dispatch.get():
            return None
        return _RequestProbe(self, "tool", {
            "api_request_id": str(api_request_id or ""),
            "tool_call_id": opaque_tool_call_id(tool_call_id or ""),
            "tool_name": str(tool_name or ""),
        })

    def mark_inexact(self, reason: str) -> None:
        if reason not in _INEXACT_REASONS:
            raise ValueError(f"unsupported execution accounting reason: {reason}")
        with self._lock:
            if not self._sealed:
                self._inexact.add(reason)

    def _begin_execution(self, kind: str, identity: dict[str, Any]) -> dict[str, Any]:
        event: dict[str, Any] = {
            "api_request_id": identity["api_request_id"], "execution_id": uuid.uuid4().hex,
            "status": "running", "started_at": _utc_now(), "completed_at": None,
        }
        if kind == "tool":
            event.update(tool_call_id=identity["tool_call_id"], tool_name=identity["tool_name"])
        with self._lock:
            if self._sealed:
                return event
            if kind == "provider":
                ordinal = self._attempts.get(event["api_request_id"], 0) + 1
                self._attempts[event["api_request_id"]] = ordinal
                event["attempt_ordinal"] = ordinal
                self.provider_executions.append(event)
            else:
                self.tool_executions.append(event)
        return event

    def _finish_execution(self, event: dict[str, Any], status: str) -> None:
        with self._lock:
            if event["status"] == "running":
                event["status"] = status
                event["completed_at"] = _utc_now()

    def _record_decision(self, kind: str, identity: dict[str, Any], event: Optional[dict[str, Any]],
                         provenance: Any) -> None:
        frames, exact = _sanitize_frames(provenance)
        terminal_reached = bool(getattr(provenance, "terminal_reached", event is not None))
        if event is not None:
            decision = "executed"
        elif terminal_reached:
            decision = "not_dispatched"
        else:
            decision = "short_circuited"
        entry = {**identity, "decision": decision, "terminal_reached": terminal_reached,
                 "execution_ids": [event["execution_id"]] if event is not None else [],
                 "middleware_trace": frames}
        with self._lock:
            if self._sealed:
                return
            if not exact:
                self._inexact.add("middleware_provenance_unattributed")
            if terminal_reached != (event is not None):
                self._inexact.add(f"{'provider' if kind == 'provider' else 'tool'}_execution_runner_mismatch")
            (self.request_decisions if kind == "provider" else self.tool_decisions).append(entry)

    def finalize(self, turn_id: str, final_status: str) -> Path:
        """Seal, then publish an owner-only receipt; raises FileExistsError for an existing one."""
        with self._lock:
            self._sealed = True
            prefix = f"{turn_id}:api:"
            for event in (*self.provider_executions, *self.tool_executions):
                if not str(event["api_request_id"]).startswith(prefix):
                    self._inexact.add("request_identity_mismatch")
            running = any(e["status"] == "running" for e in (*self.provider_executions, *self.tool_executions))
            if final_status == "completed" and (self._inexact or running):
                final_status = "incomplete"
            payload: dict[str, Any] = {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "receipt_id": self.receipt_id,
                "session_id": self.session_id,
                "turn_id": turn_id,
                "gateway_pid": self.gateway_pid,
                "gateway_runtime_id": self.gateway_runtime_id,
                "started_at": self.started_at,
                "completed_at": _utc_now(),
                "final_status": final_status,
                "provider_execution_count": len(self.provider_executions),
                "tool_execution_count": len(self.tool_executions),
                "execution_accounting": {
                    "status": "inexact" if self._inexact else "exact",
                    "inexact_reasons": sorted(self._inexact),
                },
                "request_decisions": copy.deepcopy(self.request_decisions),
                "tool_decisions": copy.deepcopy(self.tool_decisions),
                "provider_executions": copy.deepcopy(self.provider_executions),
                "tool_executions": copy.deepcopy(self.tool_executions),
            }
        canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["receipt_digest"] = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
        encoded = (json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode("utf-8")
        path = get_turn_execution_receipt_path(self.session_id, turn_id)
        _write_receipt(path, encoded)
        return path


def mark_active_turn_inexact(reason: str) -> None:
    receipt = _active_receipt.get()
    if receipt is not None:
        receipt.mark_inexact(reason)


def _final_status(result: Any, error: Optional[BaseException]) -> str:
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    if error is not None:
        return "failed"
    if isinstance(result, dict):
        if result.get("interrupted"):
            return "interrupted"
        if result.get("failed"):
            return "failed"
        if result.get("completed"):
            return "completed"
    return "incomplete"


def run_turn_without_receipt(run: Callable[[], Any]) -> Any:
    """Run a turn with receipts off; shadows an outer receipt so a nested turn cannot write into it."""
    if _active_receipt.get() is None:
        return run()
    token = _active_receipt.set(None)
    try:
        return run()
    finally:
        _active_receipt.reset(token)


def run_turn_with_receipt(agent: Any, task_id: Optional[str], run: Callable[[], Any]) -> Any:
    """Run one turn under a fresh receipt and finalize it on every exit path."""
    session_id = str(getattr(agent, "session_id", "") or "")
    prior_turn_id = getattr(agent, "_current_turn_id", None)
    gateway_pid, gateway_runtime_id = _process_identity()
    receipt = TurnExecutionReceipt(session_id, gateway_pid=gateway_pid, gateway_runtime_id=gateway_runtime_id)
    token = _active_receipt.set(receipt)
    result: Any = None
    error: Optional[BaseException] = None
    try:
        result = run()
        return result
    except BaseException as exc:
        error = exc
        raise
    finally:
        _active_receipt.reset(token)
        bound = getattr(agent, "_current_turn_id", None)
        # A turn that failed before it bound its own id must not reuse the previous turn's id.
        turn_id = str(bound) if bound and bound != prior_turn_id else (
            f"{session_id or 'session'}:{task_id or uuid.uuid4()}:{uuid.uuid4().hex[:8]}")
        try:
            receipt.finalize(turn_id, _final_status(result, error))
        except Exception:
            logger.warning("Failed to finalize turn execution receipt", exc_info=True)


__all__ = [
    "RECEIPT_SCHEMA_VERSION",
    "TurnExecutionReceipt",
    "active_turn_receipt",
    "execution_receipts_enabled",
    "get_turn_execution_receipt_dir",
    "get_turn_execution_receipt_path",
    "mark_active_turn_inexact",
    "opaque_tool_call_id",
    "run_turn_with_receipt",
    "run_turn_without_receipt",
]
