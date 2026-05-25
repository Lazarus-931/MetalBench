"""Verifier — third stage of the Agent Steel pipeline.

Job: apply the Implementor's diff to disk, rebuild metallibs, run
`./bench` N times, gate on correctness + median-drop, then KEEP or REVERT
the change. Writes one AttemptEntry to the history DB either way.

The Verifier is the *only* stage that mutates the working tree. Profiler
and Implementor are pure functions — they take a kernel name and produce
data. The Verifier closes the loop.
"""
from .agent import VerifierAgent, VerifierResult, verify

__all__ = ["VerifierAgent", "VerifierResult", "verify"]
