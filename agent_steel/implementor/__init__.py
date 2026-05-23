"""Implementor — produce a concrete .metal diff from a ProfilerReport.

This is the second stage of the Agent Steel pipeline:

    Profiler → Implementor → Verifier → (Loop)

The Implementor never writes to disk. It generates a unified diff and the
would-be-modified source; the Verifier handles writing, building, and
benching the candidate. That separation makes every attempt rollback-able
in one line if it regresses.
"""
from .agent import ImplementorAgent, ImplementorResult, implement

__all__ = ["ImplementorAgent", "ImplementorResult", "implement"]
