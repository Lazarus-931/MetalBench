"""Greedy search strategy — one candidate per round, terminate on plateau."""
from __future__ import annotations
from dataclasses import dataclass

from ..history import AttemptEntry


@dataclass
class GreedyStrategy:
    """Single-worker greedy strategy.

    Always picks the top untried Candidate. Terminates when any of:
    - `max_rounds` rounds have been attempted
    - the kernel reaches `sol_target` (default 0.85 — near-optimal)
    - `max_no_improvement` consecutive rounds produced no kept attempt
    - the Optimizer returns zero candidates (exhausted)
    """

    max_rounds: int = 10
    max_no_improvement: int = 3
    sol_target: float = 0.85

    def should_terminate(
        self,
        round_num: int,
        history: list[AttemptEntry],
        last_sol: float,
    ) -> tuple[bool, str]:
        """Return (terminate?, reason)."""
        if round_num >= self.max_rounds:
            return True, f"max_rounds={self.max_rounds} reached"
        if last_sol >= self.sol_target:
            return True, f"sol={last_sol*100:.0f}% ≥ {self.sol_target*100:.0f}% target"
        # Count trailing non-kept attempts
        non_kept_streak = 0
        for entry in reversed(history):
            if entry.kept:
                break
            non_kept_streak += 1
        if non_kept_streak >= self.max_no_improvement:
            return True, f"{self.max_no_improvement} rounds without improvement"
        return False, ""
