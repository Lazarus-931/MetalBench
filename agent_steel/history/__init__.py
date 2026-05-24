"""Persistent attempt history for Agent Steel.

Each (kernel, chip) pair gets a JSONL log at
`.agent-steel/history/<kernel>__<chip>.jsonl`. The Verifier writes one
`AttemptEntry` per round; the Implementor reads prior entries to filter
its strategy selection (no re-trying failed techniques).

Append-only — no edits or deletions. Robust against crashes, easy to grep.
"""
from .models import AttemptEntry
from .db import AttemptDB

__all__ = ["AttemptEntry", "AttemptDB"]
