"""Optimizer — pick the next technique to try on a kernel.

Two-stage split per agent_steel design:

    Profiler  → Optimizer  →  Implementor  →  Verifier  →  (Loop)
    (diagnose) (extract)     (codegen)        (apply+gate)

The Optimizer is deterministic: it queries `patterns.json`, the
`ProfilerReport`'s LLM-driven `suggested_edits`, and the per-kernel history
DB, then ranks. No LLM call in the extract path. The Implementor (next
stage) is the only LLM-driven step in the loop.

Public API:

    from agent_steel.optimizer import OptimizerAgent, extract, Candidate

    opt = OptimizerAgent()
    candidates = opt.run(profiler_report)
    top = candidates[0]
"""
from .agent import OptimizerAgent
from .extraction import Candidate, extract

__all__ = ["OptimizerAgent", "Candidate", "extract"]
