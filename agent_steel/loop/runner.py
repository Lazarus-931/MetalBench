"""Loop runner — orchestrates Profiler → Optimizer → Implementor → Verifier."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..history import AttemptDB, AttemptEntry
from ..implementor import implement
from ..optimizer import extract, Candidate
from ..profiler import profile, ProfilerReport, SuggestedEdit
from ..providers import Provider
from ..verifier import verify, VerifierResult
from .greedy import GreedyStrategy


REPO = Path(__file__).resolve().parents[2]


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


def _candidate_to_suggested_edit(c: Candidate) -> SuggestedEdit:
    """Coerce a Candidate into the SuggestedEdit shape the Implementor consumes."""
    return SuggestedEdit(
        technique=c.technique,
        rationale=c.rationale,
        target_lines=c.target_lines or "see code analysis",
        expected_impact=c.expected_impact or (
            f"prior wins on: {', '.join(w.get('kernel','?') for w in c.wins_on)}"
            if c.wins_on else "no prior wins recorded"
        ),
    )


def _chip_id_from_bench(chip_str: str | None) -> str:
    from agent_steel.chips import detect_generation
    return f"apple-{detect_generation(chip_str, fallback='m2')}"


def _set_for_kernel(kernel: str) -> str:
    """Return common/standard/full for the kernel name (best-effort)."""
    for s in ("common", "standard", "full"):
        d = REPO / "metal" / "kernels" / s
        if not d.is_dir():
            continue
        if (d / f"{kernel}.metal").is_file() or (d / kernel).is_dir():
            return s
    return "common"


# ---------------------------------------------------------------------------
# Main entry.
# ---------------------------------------------------------------------------

def run_loop(
    kernel: str,
    *,
    provider: Provider,
    strategy: GreedyStrategy | None = None,
    db: AttemptDB | None = None,
    chip_override: str | None = None,
) -> LoopResult:
    """Run the full optimization loop on a kernel.

    Each round:
      1. Profile (re-bench, get fresh roofline + source analysis)
      2. Extract candidates (patterns + profiler.suggested_edits + history)
      3. Implementor builds a diff for the top untried candidate
      4. Verifier applies, benches, gates, logs

    Returns a LoopResult summarizing every attempt.
    """
    strategy = strategy or GreedyStrategy()
    db = db or AttemptDB()
    set_name = _set_for_kernel(kernel)

    # Initial bench — establishes the chip + the baseline before_ms.
    initial_report = profile(kernel, provider=provider, skip_llm=True)
    chip = chip_override or _chip_id_from_bench(initial_report.chip)
    initial_ms = (initial_report.packet.get("timing_trust") or {}).get("median_ms")
    initial_sol = initial_report.sol

    history: list[AttemptEntry] = list(db.read(kernel, chip))
    best_ms = initial_ms
    best_parent_id: str | None = None
    rounds_run = 0
    kept_count = 0
    term_reason = ""

    for round_num in range(strategy.max_rounds):
        rounds_run = round_num + 1

        # ----- 1. Profile -----
        # Use provider on round > 0 so we get LLM suggested_edits when the
        # pattern store is exhausted. Round 0 already happened above.
        if round_num == 0:
            report = initial_report
        else:
            report = profile(kernel, provider=provider, skip_llm=False)

        # Termination check pre-extract — if SOL is already at target, stop.
        terminate, reason = strategy.should_terminate(
            round_num, history, report.sol
        )
        if terminate:
            term_reason = reason
            break

        # ----- 2. Extract candidates -----
        candidates = extract(report, chip=chip, db=db)
        if not candidates:
            # If the LLM was skipped on this round, `report.suggested_edits`
            # is empty and we relied entirely on patterns.json. If patterns
            # matched nothing either, the user has two recoverable options:
            # seed patterns.json for this (bottleneck, kernel-kind) bucket,
            # or re-run with the LLM enabled (omit --no-llm). Spell that
            # out instead of an opaque "no candidates" message.
            llm_was_off = (round_num == 0)  # round 0 always runs skip_llm=True
            if llm_was_off and not report.suggested_edits:
                term_reason = (
                    "LLM was off on round 0 (skip_llm=True) AND patterns.json "
                    f"had no matching entry for bottleneck "
                    f"{report.bottleneck_class!r} on kernel {report.kernel!r}. "
                    "Either seed patterns.json for this bucket or re-run "
                    "with a provider so the profiler can generate "
                    "kernel-specific suggested_edits."
                )
            else:
                term_reason = (
                    f"no candidates returned from optimizer "
                    f"(patterns + profiler.suggested_edits both empty for "
                    f"bottleneck={report.bottleneck_class!r})"
                )
            break

        # Pick top candidate; the Implementor will further filter by prior_attempts.
        prior_techniques = db.techniques_tried(kernel, chip)
        top: Candidate | None = None
        for c in candidates:
            t = c.technique.lower()
            if any(t in p.lower() or p.lower() in t for p in prior_techniques):
                continue
            top = c
            break
        if top is None:
            term_reason = "all candidates already tried"
            break

        # Surface the selected candidate in the report so the Implementor uses it.
        report.suggested_edits = [_candidate_to_suggested_edit(top)] + report.suggested_edits

        # ----- 3. Implementor -----
        impl = implement(
            report,
            provider=provider,
            prior_attempts=prior_techniques,
            set_name=set_name,
        )

        # ----- 4. Verifier -----
        result = verify(
            impl,
            chip=chip, set_name=set_name,
            before_ms=best_ms,
            db=db,
            parent_id=best_parent_id,
            generation=round_num,
        )

        # Refresh local history slice and update best pointer.
        history = list(db.read(kernel, chip))
        if result.kept and result.after_ms is not None:
            kept_count += 1
            best_ms = result.after_ms
            best_parent_id = result.attempt_id

        # Post-round termination check.
        terminate, reason = strategy.should_terminate(
            rounds_run, history, report.sol
        )
        if terminate:
            term_reason = reason
            break

    if not term_reason:
        term_reason = f"completed {rounds_run} rounds"

    overall_imp = None
    if initial_ms and best_ms and initial_ms > 0:
        overall_imp = (initial_ms - best_ms) / initial_ms * 100.0

    return LoopResult(
        kernel=kernel, chip=chip,
        rounds_run=rounds_run, kept_attempts=kept_count,
        initial_ms=initial_ms, best_ms=best_ms,
        overall_improvement_pct=overall_imp,
        termination_reason=term_reason,
        attempts=history,
    )
