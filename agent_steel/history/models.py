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

    id: str = field(default_factory=_gen_id)
    timestamp: str = field(default_factory=_now)
    kernel: str = ""
    chip: str = ""
    parent_id: str | None = None
    generation: int = 0

    technique: str = ""
    diff: str = ""
    files_touched: list[str] = field(default_factory=list)
    source_snapshot: str = ""
    gputrace_metrics: dict[str, Any] = field(default_factory=dict)

    correctness_passed: bool = False
    max_err: float | None = None
    before_ms: float | None = None
    after_ms: float | None = None
    improvement_pct: float | None = None
    runs_ms: list[float] = field(default_factory=list)
    stability_cv: float | None = None

    kept: bool = False
    rollback_reason: str = ""

    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
