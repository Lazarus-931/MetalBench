"""Optimizer agent — thin wrapper over extract().

The Optimizer in Agent Steel is split into:
- extract() — deterministic, queries patterns.json + profiler.suggested_edits +
  history DB. Returns ranked Candidates.
- The Implementor (separate module) — generates the actual .metal diff for
  the top Candidate.

This split costs zero LLM calls for the extraction step. If/when patterns.json
grows past ~30-50 entries we can layer embedding-based retrieval on top
without changing the public API.
"""
from __future__ import annotations
from typing import Any

from ..history import AttemptDB
from ..profiler import ProfilerReport
from .extraction import Candidate, extract


class OptimizerAgent:
    """Pick the next technique to try on a kernel.

    Usage:
        opt = OptimizerAgent()
        candidates = opt.run(profiler_report)
        top = candidates[0]  # → hand to Implementor
    """

    def __init__(self, db: AttemptDB | None = None):
        self.db = db or AttemptDB()

    def run(self, report: ProfilerReport, *, chip: str | None = None) -> list[Candidate]:
        return extract(report, chip=chip, db=self.db)
