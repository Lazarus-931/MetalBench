"""Optimizer — LLM agent that writes the next iteration of a kernel.

    optimize(profile, provider=...) -> OptimizerResult

Inputs: Profiler narrative + AttemptDB log + current .metal + MLX reference.
Writes the new candidate IN-PLACE to metal/kernels/<set>/<kernel>.metal.
Accuracy is gated here; performance is gated by the Verifier (against
session.json's leaderboard, not local history).
"""
from .agent import OptimizerAgent, OptimizerResult, optimize

__all__ = ["OptimizerAgent", "OptimizerResult", "optimize"]
