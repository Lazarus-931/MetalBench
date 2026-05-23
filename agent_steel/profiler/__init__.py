"""Profiler — diagnostic stage of the Agent Steel pipeline.

All profiler concerns live here:

    agent.py          Profiler agent: bench → roofline → LLM-mediated diagnosis
    bench_runner.py   Run ./bench and parse its human-readable output
    gputrace/         Parser for Apple .gputrace bundles (optional context)

Public API:

    from agent_steel.profiler import profile, ProfilerReport
    from agent_steel.profiler.gputrace import parse as parse_gputrace
"""
from .agent import (
    ProfilerReport,
    SuggestedEdit,
    profile,
)
from .bench_runner import BenchResult, run_bench

__all__ = [
    "BenchResult",
    "ProfilerReport",
    "SuggestedEdit",
    "profile",
    "run_bench",
]
