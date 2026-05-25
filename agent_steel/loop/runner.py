"""Loop runner — orchestrates Profiler → Optimizer → Verifier.

Three agents, two LLM:

    Profiler  (LLM)   : .gputrace + bench  → 2-3 paragraph narrative
    Optimizer (LLM)   : narrative + attempt log + .metal + MLX → new .metal
                        accuracy gate (correctness ≥ 99% via ./bench)
    Verifier  (rules) : bench×N, perf gate, log ±Δ% to AttemptDB

The Loop owns chip identification, gputrace path resolution, prior-best source
caching for revert, and per-round greedy termination.
"""
from __future__ import annotations
import fcntl
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from ..history import AttemptDB, AttemptEntry
from ..optimizer import OptimizerResult, optimize
from ..profiler import ProfileResult, profile
from ..providers import Provider
from ..verifier import verify
from .greedy import GreedyStrategy

REPO = Path(__file__).resolve().parents[2]
GPUTRACE_DIR = REPO / "results"
_SESSION_LOCK_DIR = Path(os.environ.get(
    "AGENT_STEEL_SESSION_LOCK_DIR",
    str(Path.home() / ".agent-steel" / "locks"),
))


# Two agent-steel processes on the same kernel × chip would race on AttemptDB
# lineage and the active .metal source. This lock serializes them.
@contextmanager
def _session_lock(kernel: str, chip: str):
    _SESSION_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    safe = f"{kernel}__{chip.replace('/', '_').replace(' ', '_')}.lock"
    path = _SESSION_LOCK_DIR / safe
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, f"pid={os.getpid()} t={time.time()}\n".encode())
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@dataclass
class LoopResult:
    kernel: str
    chip: str
    rounds_run: int
    kept_attempts: int
    initial_ms: float | None
    best_ms: float | None
    overall_improvement_pct: float | None
    termination_reason: str
    attempts: list[AttemptEntry] = field(default_factory=list)


def _chip_id(chip_generation: str) -> str:
    return f"apple-{chip_generation}"


def _gputrace_path(kernel: str, chip_generation: str) -> str | None:
    """Look for a .gputrace bundle under results/<chip-gen>/."""
    p = GPUTRACE_DIR / chip_generation / f"{kernel}.gputrace"
    return str(p) if p.is_dir() else None


def _sol_from_profile(prof: ProfileResult) -> float:
    m = prof.chip_aware_metrics or {}
    sol_pct = m.get("sol_compute_pct") or m.get("device_memory_bandwidth_pct") or 0.0
    return float(sol_pct) / 100.0


def _peek_generation(chip_override: str | None) -> str:
    """Best-effort chip-gen guess BEFORE the first bench, for gputrace lookup."""
    if chip_override:
        s = chip_override.lower()
        for g in ("m5", "m4", "m3", "m1"):
            if g in s:
                return g
    return "m2"


def run_loop(
    kernel: str,
    *,
    provider: Provider,
    strategy: GreedyStrategy | None = None,
    db: AttemptDB | None = None,
    chip_override: str | None = None,
) -> LoopResult:
    strategy = strategy or GreedyStrategy()
    db = db or AttemptDB()

    # Initial profile must run before the session lock — it tells us the chip id.
    initial_profile = profile(
        kernel, provider=provider,
        gputrace_path=_gputrace_path(kernel, _peek_generation(chip_override)),
    )
    chip = chip_override or _chip_id(initial_profile.chip_generation)

    with _session_lock(kernel, chip):
        return _run_loop_inner(
            kernel, chip, initial_profile, provider, strategy, db,
        )


def _run_loop_inner(
    kernel: str,
    chip: str,
    initial_profile: ProfileResult,
    provider: Provider,
    strategy: GreedyStrategy,
    db: AttemptDB,
) -> LoopResult:
    initial_ms = initial_profile.bench.kernel_ms
    best_ms = initial_ms

    from ..optimizer.agent import _resolve_metal_path
    active_metal_path, _ = _resolve_metal_path(kernel, initial_profile.chip_generation)
    prior_best_source = active_metal_path.read_text()

    existing = list(db.read(kernel, chip))
    baseline_id: str | None = next(
        (e.id for e in existing if e.technique == "baseline" and e.kept), None,
    )
    if baseline_id is None:
        baseline = AttemptEntry(
            kernel=kernel,
            chip=chip,
            technique="baseline",
            source_snapshot=prior_best_source,
            gputrace_metrics=initial_profile.chip_aware_metrics or {},
            files_touched=[str(active_metal_path.relative_to(REPO))],
            correctness_passed=initial_profile.bench.correct,
            max_err=initial_profile.bench.max_err,
            before_ms=None,
            after_ms=initial_ms,
            improvement_pct=None,
            kept=True,
            generation=0,
            notes="session baseline — captured immediately after initial profile",
        )
        db.append(baseline)
        baseline_id = baseline.id

    rounds_run = 0
    kept_count = 0
    term_reason = ""

    prof = initial_profile
    for round_num in range(strategy.max_rounds):
        rounds_run = round_num + 1

        if round_num > 0:
            prof = profile(
                kernel, provider=provider,
                gputrace_path=_gputrace_path(kernel, prof.chip_generation),
            )

        history = list(db.read(kernel, chip))
        terminate, reason = strategy.should_terminate(
            round_num, history, _sol_from_profile(prof),
        )
        if terminate:
            term_reason = reason
            break

        opt: OptimizerResult = optimize(prof, provider=provider, db=db, chip_id=chip)
        if not opt.accuracy_passed:
            continue

        ver = verify(
            kernel, chip=chip, db=db,
            technique_summary=opt.change_summary,
            revert_source=prior_best_source,
            gputrace_metrics=prof.chip_aware_metrics,
        )

        if ver.kept and ver.after_ms is not None:
            kept_count += 1
            best_ms = ver.after_ms
            prior_best_source = active_metal_path.read_text()

        terminate, reason = strategy.should_terminate(
            rounds_run, list(db.read(kernel, chip)), _sol_from_profile(prof),
        )
        if terminate:
            term_reason = reason
            break

    if not term_reason:
        term_reason = f"completed {rounds_run} rounds"

    overall = None
    if initial_ms and best_ms and initial_ms > 0:
        overall = (initial_ms - best_ms) / initial_ms * 100.0

    return LoopResult(
        kernel=kernel, chip=chip,
        rounds_run=rounds_run, kept_attempts=kept_count,
        initial_ms=initial_ms, best_ms=best_ms,
        overall_improvement_pct=overall,
        termination_reason=term_reason,
        attempts=list(db.read(kernel, chip)),
    )
