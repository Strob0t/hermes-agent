"""Opt-in per-turn BM25 skill retrieval (fork finding #118).

Ranks the loadable skills against the current user message with the same BM25
scorer the tool_search catalog uses (``tools.tool_search_catalog._bm25_score``),
so there is no second BM25 implementation, no embedding index, no SQLite and no
RRF fusion. The ranked top-k names and descriptions become a short hint on the
API copy of the turn's user message (see ``agent.turn_context``); the stored
message and the system-prompt skill index stay byte-identical.

Settings (read once at agent init, never per turn):
  skills.retrieval.enabled  — default False (opt-in)
  skills.retrieval.top_k    — default 3, clamped to 1..5
Env fallback, only when the config key is absent (config wins):
  HERMES_SKILL_RETRIEVAL=on|off, HERMES_SKILL_RETRIEVAL_TOP_K=<int>
"""
from __future__ import annotations

import logging
import os
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.tool_search_catalog import _bm25_score, _tokenize

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3
MIN_TOP_K = 1
MAX_TOP_K = 5

ENV_ENABLED = "HERMES_SKILL_RETRIEVAL"
ENV_TOP_K = "HERMES_SKILL_RETRIEVAL_TOP_K"


def clamp_top_k(value: Any, default: int = DEFAULT_TOP_K) -> int:
    """Coerce a configured/env top_k to an int inside [1, 5]; bad values use *default*."""
    try:
        if isinstance(value, bool):
            raise ValueError(value)
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        parsed = default
    return max(MIN_TOP_K, min(MAX_TOP_K, parsed))


def _env_enabled() -> bool:
    return (os.environ.get(ENV_ENABLED) or "").strip().lower() == "on"


def resolve_retrieval_settings(agent_cfg: Any) -> tuple[bool, int]:
    """(enabled, top_k) from the loaded config's ``skills.retrieval`` section.

    ``load_config()`` merges DEFAULT_CONFIG, so an explicit config value always wins.
    The env vars are a fallback for configs without the key (older/partial dicts, or a
    config that failed to load): only then do HERMES_SKILL_RETRIEVAL(_TOP_K) apply.
    """
    skills_cfg = agent_cfg.get("skills") if isinstance(agent_cfg, dict) else None
    retrieval = skills_cfg.get("retrieval") if isinstance(skills_cfg, dict) else None
    if not isinstance(retrieval, dict):
        return _env_enabled(), clamp_top_k(os.environ.get(ENV_TOP_K))
    enabled = bool(retrieval.get("enabled", False)) if "enabled" in retrieval else _env_enabled()
    top_k = (clamp_top_k(retrieval["top_k"]) if "top_k" in retrieval
             else clamp_top_k(os.environ.get(ENV_TOP_K)))
    return enabled, top_k


def apply_retrieval_settings(agent: Any, agent_cfg: Any) -> None:
    """Snapshot the settings on the agent at init; turns never re-read config or env."""
    enabled, top_k = resolve_retrieval_settings(agent_cfg)
    agent._skills_retrieval_enabled = enabled
    agent._skills_retrieval_top_k = top_k


@dataclass(frozen=True)
class RetrievedSkill:
    name: str
    description: str
    score: float


@dataclass
class _Index:
    entries: list[tuple[str, str]]  # (name, description)
    doc_tokens: list[list[str]]
    doc_lengths: list[int]
    avg_dl: float
    doc_freq: dict[str, int]

    def rank(self, query: str, top_k: int) -> list[RetrievedSkill]:
        if not self.entries or not query:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        n_docs = len(self.entries)
        scored = [
            (i, _bm25_score(query_tokens, tokens, self.doc_lengths, self.avg_dl,
                            self.doc_freq, n_docs))
            for i, tokens in enumerate(self.doc_tokens)
        ]
        # Positive scores only: an unranked filler hint is noise, not retrieval.
        # Name tie-break keeps equal scores deterministic across runs.
        scored.sort(key=lambda pair: (-pair[1], self.entries[pair[0]][0]))
        return [
            RetrievedSkill(self.entries[i][0], self.entries[i][1], score)
            for i, score in scored[:max(MIN_TOP_K, top_k)] if score > 0.0
        ]


_INDEX_LOCK = threading.Lock()
_INDEX_CACHE: tuple[tuple, _Index] | None = None


def invalidate_retrieval_index() -> None:
    """Drop the cached corpus; called after skill_manage create/patch/delete."""
    global _INDEX_CACHE
    with _INDEX_LOCK:
        _INDEX_CACHE = None


def _tags_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [part.strip().strip("\"'") for part in text.split(",") if part.strip()]


def _scan_entries() -> list[tuple[tuple, tuple[str, str, str]]]:
    """Visible skills with their BM25 text: (cache identity, (name, description, search text)).

    Visibility/dedup come from the skills_list catalog, so a hidden, shadowed or disabled
    skill is never hinted. ``triggers`` and ``tags`` are frontmatter-only fields, so they
    are read here alongside the name and description the catalog already resolved.
    """
    from tools.skills_tool import _parse_frontmatter, _read_skill_text, _skill_catalog

    scanned: list[tuple[tuple, tuple[str, str, str]]] = []
    for skill in _skill_catalog():
        path: Path = skill["path"]
        try:
            stat = path.stat()
            frontmatter, _ = _parse_frontmatter(_read_skill_text(path)[:4000])
        except Exception as exc:
            logger.debug("skill retrieval: skipping %s: %s", path, exc)
            continue
        name = str(skill.get("load_name") or skill.get("name") or path.parent.name)
        description = str(skill.get("description") or "")
        triggers = frontmatter.get("triggers") or frontmatter.get("trigger") or []
        trigger_terms = ([str(t).strip() for t in triggers if str(t).strip()]
                         if isinstance(triggers, list) else _tags_list(triggers))
        tags = _tags_list(frontmatter.get("tags"))
        search_text = " ".join([name, description, *trigger_terms, *tags])
        scanned.append(((name, str(path), stat.st_mtime_ns, stat.st_size),
                        (name, description, search_text)))
    return scanned


def get_index() -> _Index:
    """Process-local corpus, rebuilt when any skill file's identity/signature changes."""
    global _INDEX_CACHE
    scanned = _scan_entries()
    key = tuple(item[0] for item in scanned)
    with _INDEX_LOCK:
        if _INDEX_CACHE is not None and _INDEX_CACHE[0] == key:
            return _INDEX_CACHE[1]
    entries = [(name, description) for _, (name, description, _text) in scanned]
    doc_tokens = [_tokenize(text) for _, (_name, _desc, text) in scanned]
    doc_lengths = [len(tokens) for tokens in doc_tokens]
    avg_dl = sum(doc_lengths) / max(len(doc_lengths), 1)
    doc_freq = dict(Counter(token for tokens in doc_tokens for token in set(tokens)))
    index = _Index(entries=entries, doc_tokens=doc_tokens, doc_lengths=doc_lengths,
                   avg_dl=avg_dl, doc_freq=doc_freq)
    with _INDEX_LOCK:
        _INDEX_CACHE = (key, index)
    return index


def retrieve_skills(query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedSkill]:
    """Top-k skills for *query*; [] when disabled-by-content (no positive BM25 match)."""
    try:
        return get_index().rank(query, clamp_top_k(top_k))
    except Exception as exc:  # retrieval is a hint; it must never break a turn
        logger.debug("skill retrieval failed: %s", exc)
        return []
