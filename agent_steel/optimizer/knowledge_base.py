"""Knowledge base for the Optimizer — hand-curated patterns + anti-patterns.

`patterns.json` is the central store. Indexed bottleneck class → kernel kind →
list of techniques, each with a `wins_on` list naming the kernels/chips where
this exact pattern produced ≥15% improvement.

Querying is exact-match + small fuzzy logic in `extraction.py`. Embedding-
based retrieval would land later when the store passes ~30-50 entries.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PATTERNS_PATH = REPO / "agent_steel" / "optimizer" / "patterns.json"
ANTI_PATTERNS_PATH = REPO / "agent_steel" / "anti_patterns.md"


def _load_patterns() -> dict[str, Any]:
    if not PATTERNS_PATH.is_file():
        return {"_meta": {"version": 0}}
    return json.loads(PATTERNS_PATH.read_text())


def _load_anti_patterns() -> str:
    if not ANTI_PATTERNS_PATH.is_file():
        return ""
    return ANTI_PATTERNS_PATH.read_text()


# Module-level cache — patterns.json is small (~5 KB) and read-only at runtime.
_PATTERNS_CACHE: dict[str, Any] | None = None
_ANTI_CACHE: str | None = None


def patterns() -> dict[str, Any]:
    global _PATTERNS_CACHE
    if _PATTERNS_CACHE is None:
        _PATTERNS_CACHE = _load_patterns()
    return _PATTERNS_CACHE


def anti_patterns_text() -> str:
    global _ANTI_CACHE
    if _ANTI_CACHE is None:
        _ANTI_CACHE = _load_anti_patterns()
    return _ANTI_CACHE


# ---------------------------------------------------------------------------
# Querying.
# ---------------------------------------------------------------------------

def lookup(
    bottleneck_class: str,
    kernel_kind: str,
) -> list[dict[str, Any]]:
    """Return the list of patterns for (bottleneck, kernel_kind).

    Falls back gracefully:
    - exact (bottleneck, kernel_kind) match first
    - then (bottleneck, "any") if a "any" bucket exists
    - then empty list
    """
    p = patterns()
    by_bottleneck = p.get(bottleneck_class) or {}
    if not isinstance(by_bottleneck, dict):
        return []
    matched: list[dict[str, Any]] = list(by_bottleneck.get(kernel_kind) or [])
    matched += list(by_bottleneck.get("any") or [])
    return matched


def all_techniques_for_bottleneck(bottleneck_class: str) -> list[dict[str, Any]]:
    """All patterns under a bottleneck class, across kernel kinds.

    Used when the kernel kind is ambiguous and we want to surface the
    full menu of plausible techniques.
    """
    p = patterns()
    by_bottleneck = p.get(bottleneck_class) or {}
    out: list[dict[str, Any]] = []
    for kind, items in by_bottleneck.items():
        if isinstance(items, list):
            out.extend(items)
    return out
