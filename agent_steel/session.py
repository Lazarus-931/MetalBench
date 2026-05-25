"""Read MetalBench's session.json (the leaderboard) — the source of truth for
per-(kernel, chip) best times. Agent Steel benches against this, not its own
local history."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SESSION_PATH = REPO / "session.json"

CALIBRATION_TOL = 0.03  # ±3% — see docs/calibration.md


def leaderboard_best_ms(kernel: str, chip: str) -> float | None:
    """Return session.json[chip][kernel].best_time_ms, or None if no record exists."""
    if not SESSION_PATH.is_file():
        return None
    try:
        s = json.loads(SESSION_PATH.read_text())
    except json.JSONDecodeError:
        return None
    entry = (s.get(chip) or {}).get(kernel)
    return entry.get("best_time_ms") if isinstance(entry, dict) else None


def calibration_check(
    local_ms: float, leaderboard_ms: float | None, tol: float = CALIBRATION_TOL,
) -> tuple[bool, float | None, str]:
    """Confirm the local box's perf regime matches the leaderboard's within tol.

    Returns (ok, diff_pct, reason). When no leaderboard entry exists, returns
    (True, None, "no_leaderboard_entry") — the local bench becomes v0.
    """
    if leaderboard_ms is None:
        return True, None, "no_leaderboard_entry"
    if leaderboard_ms <= 0:
        return False, None, "invalid leaderboard_ms"
    diff_pct = abs(local_ms - leaderboard_ms) / leaderboard_ms
    if diff_pct <= tol:
        return True, diff_pct, "calibrated"
    return False, diff_pct, f"diff {diff_pct*100:.1f}% > {tol*100:.0f}%"
