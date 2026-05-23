"""gputrace: best-effort parser for Apple .gputrace bundles.

Reverse-engineered from observation of macOS 15/26 + Xcode 16/17 captures.
Target = compute-only subset. See the project notes for confidence levels.

Top-level entry: ``parse(path) -> dict``.

Key finding: a ``.gputrace`` is a *command-intent recording*, not a profile
log. The bundle has NO per-dispatch GPU timestamps, NO counter samples, NO
occupancy info — Xcode reconstructs that data by replaying the captured
commands. Use this parser for "what was dispatched and how"; pair it with
the live timing harness (``metal/scripts/timing.mm``) for "how fast".
"""
from .bundle import parse

__all__ = ["parse"]
