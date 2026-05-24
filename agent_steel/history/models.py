"""Persisted attempt-record models."""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class AttemptEntry:
    """One attempt by the agent loop on one (kernel, chip).

    Append-only — every attempt becomes one line in the JSONL log,
    regardless of outcome. Lets future agents reason about what was
    tried and what worked.
    """

    # Identity / lineage
    id: str = field(default_factory=_gen_id)
    timestamp: str = field(default_factory=_now)
    kernel: str = ""
    chip: str = ""                          # e.g. "apple-m2"
    parent_id: str | None = None            # the attempt whose .metal this diff was applied to
    generation: int = 0                     # depth in the lineage chain

    # What we tried
    technique: str = ""                     # human label, e.g. "loop reorder Phase A"
    diff: str = ""                          # the unified diff (kept compact when possible)
    files_touched: list[str] = field(default_factory=list)

    # Outcome
    correctness_passed: bool = False
    max_err: float | None = None
    before_ms: float | None = None
    after_ms: float | None = None
    improvement_pct: float | None = None    # ((before - after) / before) * 100
    runs_ms: list[float] = field(default_factory=list)  # individual run medians (for stability check)
    stability_cv: float | None = None       # coefficient of variation across runs

    # Decision
    kept: bool = False                      # did the Verifier accept the change?
    rollback_reason: str = ""               # if not kept, why ("<5%", "correctness", "compile_fail", "noise")

    # Free-form notes the agent or human added
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
