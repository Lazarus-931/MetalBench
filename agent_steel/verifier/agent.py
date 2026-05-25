"""Verifier agent — deterministic perf-gate. No LLM.

The Optimizer has already promoted a candidate to metal/kernels/<set>/<kernel>.metal
and passed the accuracy gate. The Verifier:

1. benches the current (promoted) kernel N times,
2. compares the median to the prior best in AttemptDB,
3. logs the ±Δ% to AttemptDB regardless of direction,
4. updates the "best" pointer if Δ is an improvement,
5. on regression, restores the prior-best .metal source (if supplied).

No LLM. No diff application. No suggestion ranking. Just rules.
"""
from __future__ import annotations
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from ..history import AttemptDB, AttemptEntry
from ..profiler import run_bench
from ..session import leaderboard_best_ms

REPO = Path(__file__).resolve().parents[2]

PRIMARY_WARMUP = 30
PRIMARY_ITERS = 100
SUB_RES_THRESHOLD_MS = 0.001

KEEP_AUTO_THRESHOLD_PCT = 5.0
REGRESSION_THRESHOLD_PCT = 5.0


@dataclass
class VerifierResult:
    kernel: str
    chip: str
    before_ms: float | None
    after_ms: float | None
    improvement_pct: float | None
    runs_ms: list[float] = field(default_factory=list)
    stability_cv: float | None = None
    kept: bool = False
    reverted: bool = False
    decision_reason: str = ""
    attempt_id: str = ""


def _bench_avg(
    kernel: str, warmup: int, iters: int, save: bool = True,
) -> tuple[float | None, float | None, float | None, bool, float | None, bool]:
    """Run a single ./bench with the configured warmup + iters; return
    (mean_ms, median_ms, min_ms, correct, max_err, sub_resolution).

    save=True lets ./bench update session.json when the run beats the
    recorded best — so a successful Verifier round automatically promotes
    its kernel to the leaderboard.
    """
    try:
        r = run_bench(kernel, warmup=warmup, iters=iters, save=save)
    except Exception:
        return None, None, None, False, None, False
    correct = bool(r.correct)
    max_err = r.max_err
    sub_res = (r.kernel_ms is None) or (r.kernel_ms <= SUB_RES_THRESHOLD_MS)
    return r.kernel_ms_mean, r.kernel_ms, r.kernel_ms_min, correct, max_err, sub_res


def _improvement_pct(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before <= 0:
        return None
    return (before - after) / before * 100.0


def _stability_cv(runs: list[float]) -> float | None:
    if len(runs) < 2:
        return None
    m = statistics.mean(runs)
    if m <= 0:
        return None
    return statistics.stdev(runs) / m


def _prior_best_ms(db: AttemptDB, kernel: str, chip: str) -> float | None:
    """Return the session.json leaderboard best (the public record). Falls back
    to the local AttemptDB best ONLY when the leaderboard has no entry for this
    (kernel, chip) — first-time benching."""
    lb = leaderboard_best_ms(kernel, chip)
    if lb is not None:
        return lb
    best = db.best(kernel, chip)
    return best.after_ms if best is not None else None


def gate(
    *,
    before_ms: float | None,
    after_ms: float | None,
    sub_resolution: bool,
    correctness_passed: bool,
) -> tuple[bool, str]:
    """Pure rules. Return (kept, reason)."""
    if not correctness_passed:
        return False, "correctness_failed"
    if sub_resolution:
        return False, "sub_resolution_unreliable"
    if before_ms is None:
        return True, "no_prior_baseline"
    imp = _improvement_pct(before_ms, after_ms)
    if imp is None:
        return False, "bench_failed"
    if imp >= KEEP_AUTO_THRESHOLD_PCT:
        return True, f"kept_+{imp:.1f}%"
    if imp <= -REGRESSION_THRESHOLD_PCT:
        return False, f"reverted_-{abs(imp):.1f}%"
    return True, f"kept_neutral_{imp:+.1f}%"


def verify(
    kernel: str,
    *,
    chip: str,
    db: AttemptDB | None = None,
    technique_summary: str = "",
    revert_source: str | None = None,
    gputrace_metrics: dict | None = None,
) -> VerifierResult:
    """Bench the promoted kernel, gate it, log to AttemptDB.

    `revert_source` is the .metal content to restore if the gate decides revert.
    The Loop passes the prior-best content here. If None and the gate decides
    revert, the regression is logged but no automatic source revert happens.
    """
    db = db or AttemptDB()
    before_ms = _prior_best_ms(db, kernel, chip)

    mean_ms, median_ms, min_ms, correct, max_err, sub_res = _bench_avg(
        kernel, warmup=PRIMARY_WARMUP, iters=PRIMARY_ITERS,
    )
    after_ms = mean_ms
    runs = [v for v in (median_ms, mean_ms, min_ms) if v is not None]
    cv = _stability_cv(runs) if len(runs) >= 2 else None

    kept, reason = gate(
        before_ms=before_ms,
        after_ms=after_ms,
        sub_resolution=sub_res,
        correctness_passed=correct,
    )
    reverted = (not kept) and (before_ms is not None)
    improvement = _improvement_pct(before_ms, after_ms)

    from ..optimizer.agent import _resolve_metal_path
    chip_gen = chip.replace("apple-", "").split("-")[0]
    active_path, _ = _resolve_metal_path(kernel, chip_gen)

    if reverted and revert_source is not None:
        active_path.write_text(revert_source)

    source_now = active_path.read_text()

    entry = AttemptEntry(
        kernel=kernel,
        chip=chip,
        technique=technique_summary[:200] or "(no summary)",
        source_snapshot=source_now if kept else "",
        gputrace_metrics=gputrace_metrics or {},
        correctness_passed=correct,
        max_err=max_err,
        before_ms=before_ms,
        after_ms=after_ms,
        improvement_pct=improvement,
        runs_ms=runs,
        stability_cv=cv,
        kept=kept,
        rollback_reason="" if kept else reason,
        notes=f"verifier: {reason}",
    )
    db.append(entry)

    return VerifierResult(
        kernel=kernel,
        chip=chip,
        before_ms=before_ms,
        after_ms=after_ms,
        improvement_pct=improvement,
        runs_ms=runs,
        stability_cv=cv,
        kept=kept,
        reverted=reverted,
        decision_reason=reason,
        attempt_id=entry.id,
    )


class VerifierAgent:
    """Deterministic verifier — non-LLM. Own dir for pipeline symmetry."""

    def __init__(self, db: AttemptDB | None = None):
        self.db = db or AttemptDB()

    def run(
        self,
        kernel: str,
        *,
        chip: str,
        technique_summary: str = "",
        revert_source: str | None = None,
        gputrace_metrics: dict | None = None,
    ) -> VerifierResult:
        return verify(
            kernel,
            chip=chip,
            db=self.db,
            technique_summary=technique_summary,
            revert_source=revert_source,
            gputrace_metrics=gputrace_metrics,
        )
