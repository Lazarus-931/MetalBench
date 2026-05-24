"""Loop — orchestrate the full Agent Steel pipeline.

    Profiler → Optimizer → Implementor → Verifier → DB → (repeat)

The Loop maintains a Strategy (currently only GreedyStrategy) that decides
when to terminate. After each kept attempt the kernel state has changed, so
the loop re-profiles and re-extracts to surface the next plausible
optimization for the new bottleneck.
"""
from .greedy import GreedyStrategy
from .runner import run_loop, LoopResult

__all__ = ["GreedyStrategy", "run_loop", "LoopResult"]
