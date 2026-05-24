"""Verifier agent — apply a diff, bench it, keep-or-revert, log the attempt.

Gate (matches the human contributor rule we've been running):

    keep if  correctness_passed AND
             median_drop_pct >= 15% AND
             passed across 5 runs

    keep with extra confirmation if  correctness_passed AND
                                     5% <= median_drop_pct < 15% AND
                                     ≥5% sustained across 10 follow-up runs

    revert otherwise.

The Verifier always writes ONE `AttemptEntry` to the history DB regardless
of outcome — that's how the loop "remembers" what's been tried.
"""
from __future__ import annotations
import subprocess
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from ..history import AttemptDB, AttemptEntry
from ..implementor import ImplementorResult
from ..profiler import run_bench

REPO = Path(__file__).resolve().parents[2]

# Default gating thresholds — keep loose-ish here; the loop can pass overrides.
KEEP_THRESHOLD_AUTO = 15.0           # ≥15% median drop → auto-keep
KEEP_THRESHOLD_CONFIRM_LOW = 5.0     # below this → revert
CONFIRMATION_RUNS = 10               # follow-up bench count for 5-15% range
PRIMARY_RUNS = 5                     # bench count for the initial gate


@dataclass
class VerifierResult:
    kernel: str
    chip: str
    technique: str

    compile_ok: bool
    correctness_passed: bool
    max_err: float | None

    before_ms: float | None
    after_ms: float | None
    improvement_pct: float | None
    runs_ms: list[float] = field(default_factory=list)
    stability_cv: float | None = None

    kept: bool = False
    rollback_reason: str = ""
    attempt_id: str = ""              # ID of the AttemptEntry we wrote


# ---------------------------------------------------------------------------
# Disk operations — apply / revert / build.
# ---------------------------------------------------------------------------

def _apply_to_disk(path: Path, applied_source: str) -> str:
    """Overwrite the file with the new source. Returns the original contents
    so the caller can revert verbatim on failure."""
    original = path.read_text()
    path.write_text(applied_source)
    return original


def _revert(path: Path, original: str) -> None:
    path.write_text(original)


def _rebuild_metallibs(timeout_s: int = 60) -> tuple[bool, str]:
    """Run `make` in the repo root. Returns (ok, stderr)."""
    try:
        proc = subprocess.run(
            ["make"], cwd=REPO, capture_output=True, text=True, timeout=timeout_s
        )
        return proc.returncode == 0, proc.stderr
    except subprocess.TimeoutExpired:
        return False, "make timed out"


# ---------------------------------------------------------------------------
# Benching helpers — wrap run_bench with run-count + stability tracking.
# ---------------------------------------------------------------------------

def _bench_n(kernel: str, n: int, iters: int = 200, warmup: int = 50) -> tuple[list[float], bool, float]:
    """Run the bench N times. Returns (median_per_run, all_correct, max_err_seen)."""
    runs: list[float] = []
    all_correct = True
    worst_err = 0.0
    for _ in range(n):
        try:
            r = run_bench(kernel, iters=iters, warmup=warmup)
        except Exception:
            return runs, False, float("inf")
        runs.append(r.kernel_ms or float("inf"))
        if not r.correct:
            all_correct = False
        if r.max_err is not None and r.max_err > worst_err:
            worst_err = r.max_err
    return runs, all_correct, worst_err


def _cv(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean = statistics.mean(xs)
    if mean <= 0:
        return None
    sd = statistics.stdev(xs)
    return sd / mean


# ---------------------------------------------------------------------------
# The actual gate.
# ---------------------------------------------------------------------------

def _evaluate_gate(
    kernel: str,
    before_ms: float | None,
    primary_runs: list[float],
) -> tuple[bool, float | None, str, list[float]]:
    """Return (kept, improvement_pct, reason, follow_up_runs)."""
    if not primary_runs:
        return False, None, "no_runs", []

    after_ms = statistics.median(primary_runs)
    if before_ms is None or before_ms <= 0:
        # No baseline → can't gate on improvement; accept correctness-only.
        return True, None, "no_baseline_kept", []

    improvement = (before_ms - after_ms) / before_ms * 100.0

    if improvement >= KEEP_THRESHOLD_AUTO:
        return True, improvement, "auto_keep_>=15%", []

    if improvement >= KEEP_THRESHOLD_CONFIRM_LOW:
        # 5-15% — need extra confirmation. Take CONFIRMATION_RUNS more benches.
        follow_up, _, _ = _bench_n(kernel, CONFIRMATION_RUNS, iters=200, warmup=50)
        if follow_up:
            sustained = statistics.median(follow_up)
            sustained_improvement = (before_ms - sustained) / before_ms * 100.0
            if sustained_improvement >= KEEP_THRESHOLD_CONFIRM_LOW:
                return True, sustained_improvement, "kept_after_confirmation", follow_up
            return False, improvement, "did_not_sustain", follow_up
        return False, improvement, "confirmation_runs_failed", []

    return False, improvement, "<5%", []


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

def verify(
    impl: ImplementorResult,
    *,
    chip: str,
    set_name: str,
    before_ms: float | None,
    db: AttemptDB | None = None,
    parent_id: str | None = None,
    generation: int = 0,
) -> VerifierResult:
    """Apply impl's diff/source to disk, bench, gate, keep-or-revert, log."""
    db = db or AttemptDB()

    # Reject early if Implementor never produced a usable result.
    if not impl.apply_succeeded or impl.applied_source is None:
        entry = AttemptEntry(
            kernel=impl.kernel, chip=chip,
            parent_id=parent_id, generation=generation,
            technique=impl.technique_attempted,
            diff=impl.diff,
            kept=False,
            rollback_reason="diff_did_not_apply",
            notes=impl.notes,
        )
        db.append(entry)
        return VerifierResult(
            kernel=impl.kernel, chip=chip, technique=impl.technique_attempted,
            compile_ok=False, correctness_passed=False, max_err=None,
            before_ms=before_ms, after_ms=None, improvement_pct=None,
            kept=False, rollback_reason="diff_did_not_apply",
            attempt_id=entry.id,
        )

    # Resolve the file we're writing to (Implementor uses the same logic;
    # we recompute here to avoid coupling).
    from ..implementor.agent import _resolve_metal_path
    metal_path = _resolve_metal_path(impl.kernel, set_name, chip)
    original_source = _apply_to_disk(metal_path, impl.applied_source)

    # Rebuild.
    compile_ok, compile_err = _rebuild_metallibs()
    if not compile_ok:
        _revert(metal_path, original_source)
        entry = AttemptEntry(
            kernel=impl.kernel, chip=chip,
            parent_id=parent_id, generation=generation,
            technique=impl.technique_attempted,
            diff=impl.diff, files_touched=impl.files_touched,
            kept=False, rollback_reason="compile_fail",
            notes=compile_err[-2000:],  # cap log noise
        )
        db.append(entry)
        return VerifierResult(
            kernel=impl.kernel, chip=chip, technique=impl.technique_attempted,
            compile_ok=False, correctness_passed=False, max_err=None,
            before_ms=before_ms, after_ms=None, improvement_pct=None,
            kept=False, rollback_reason="compile_fail",
            attempt_id=entry.id,
        )

    # Bench.
    primary, all_correct, max_err = _bench_n(impl.kernel, PRIMARY_RUNS)
    if not all_correct:
        _revert(metal_path, original_source)
        _rebuild_metallibs()  # restore the metallib too
        entry = AttemptEntry(
            kernel=impl.kernel, chip=chip,
            parent_id=parent_id, generation=generation,
            technique=impl.technique_attempted,
            diff=impl.diff, files_touched=impl.files_touched,
            correctness_passed=False, max_err=max_err,
            runs_ms=primary,
            kept=False, rollback_reason="correctness",
        )
        db.append(entry)
        return VerifierResult(
            kernel=impl.kernel, chip=chip, technique=impl.technique_attempted,
            compile_ok=True, correctness_passed=False, max_err=max_err,
            before_ms=before_ms, after_ms=None, improvement_pct=None,
            runs_ms=primary,
            kept=False, rollback_reason="correctness",
            attempt_id=entry.id,
        )

    # Gate on improvement.
    kept, improvement, reason, follow_up = _evaluate_gate(impl.kernel, before_ms, primary)
    runs_ms = primary + follow_up
    cv = _cv(runs_ms)
    after_ms = statistics.median(primary)

    if not kept:
        _revert(metal_path, original_source)
        _rebuild_metallibs()

    entry = AttemptEntry(
        kernel=impl.kernel, chip=chip,
        parent_id=parent_id, generation=generation,
        technique=impl.technique_attempted,
        diff=impl.diff, files_touched=impl.files_touched,
        correctness_passed=True, max_err=max_err,
        before_ms=before_ms, after_ms=after_ms,
        improvement_pct=improvement,
        runs_ms=runs_ms, stability_cv=cv,
        kept=kept, rollback_reason="" if kept else reason,
    )
    db.append(entry)

    return VerifierResult(
        kernel=impl.kernel, chip=chip, technique=impl.technique_attempted,
        compile_ok=True, correctness_passed=True, max_err=max_err,
        before_ms=before_ms, after_ms=after_ms,
        improvement_pct=improvement,
        runs_ms=runs_ms, stability_cv=cv,
        kept=kept, rollback_reason="" if kept else reason,
        attempt_id=entry.id,
    )


class VerifierAgent:
    """OO wrapper for symmetry with the rest of the pipeline."""

    def __init__(self, db: AttemptDB | None = None):
        self.db = db or AttemptDB()

    def run(
        self,
        impl: ImplementorResult,
        *,
        chip: str,
        set_name: str,
        before_ms: float | None = None,
        parent_id: str | None = None,
        generation: int = 0,
    ) -> VerifierResult:
        return verify(
            impl,
            chip=chip, set_name=set_name, before_ms=before_ms,
            db=self.db, parent_id=parent_id, generation=generation,
        )
