"""Optimizer — LLM agent that writes the next iteration of a kernel.

    optimize(profile, provider=...) -> OptimizerResult

Inputs: Profiler narrative + AttemptDB log + current .metal + MLX reference.
Outputs: new .metal staged at optimizer/staging/<kernel>.metal + a 2-3 sentence
change summary. Accuracy is gated here; performance is gated by the Verifier.
"""
from .agent import OptimizerAgent, OptimizerResult, STAGING_DIR, optimize

__all__ = ["OptimizerAgent", "OptimizerResult", "STAGING_DIR", "optimize"]
