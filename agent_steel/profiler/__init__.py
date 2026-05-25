"""Profiler — diagnostic stage of the Agent Steel pipeline.

    profile(kernel, provider=..., gputrace_path=...) -> ProfileResult

ProfileResult carries a 2-3 paragraph narrative + the BenchResult and
chip-aware metrics. Downstream Optimizer consumes the narrative.
"""
from .agent import ProfileResult, ProfilerAgent, profile
from .bench_runner import BenchResult, run_bench

__all__ = [
    "BenchResult",
    "ProfileResult",
    "ProfilerAgent",
    "profile",
    "run_bench",
]
