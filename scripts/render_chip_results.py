#!/usr/bin/env python3
"""Regenerate results/<chip>/results.md from session.json (single source of truth).

The website (Lazarus-931.github.io/leaderboard.html) fetches
`results/<chip>/results.md` over raw.githubusercontent.com, so this file must
stay current after agent-driven kernel updates. Per-kernel `*.json` files
under `results/<chip>/` are run artifacts; session.json holds the canonical
"best time per (chip, kernel)" record and is what we render from.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
SESSION = REPO / "session.json"


def render(chip_id: str, entries: dict) -> str:
    rows = []
    for name, e in entries.items():
        if not isinstance(e, dict):
            continue
        ms = e.get("best_time_ms")
        sp = e.get("speedup_vs_mlx")
        gflops = e.get("gflops")
        gbps = e.get("gbps")
        if ms is None:
            continue
        rows.append({
            "name": name,
            "ms": float(ms),
            "speedup": float(sp) if sp else 0.0,
            "gflops": float(gflops) if gflops else 0.0,
            "gbps": float(gbps) if gbps else 0.0,
        })
    rows.sort(key=lambda r: r["name"])
    out = [f"# {chip_id} Results", ""]
    out += ["| kernel | time (ms) | speedup | GFLOPS | GB/s |", "|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['name']} | {r['ms']:.3f} | {r['speedup']:.2f}× | {int(r['gflops'])} | {int(r['gbps'])} |"
        )
    out += ["", f"_{len(rows)} kernels._", ""]
    return "\n".join(out) + "\n"


def main() -> int:
    if not SESSION.exists():
        sys.exit(f"missing {SESSION}")
    if not RESULTS.exists():
        sys.exit(f"missing {RESULTS}")

    s = json.loads(SESSION.read_text())
    # Render one results.md per chip dir under results/. Chips without session
    # data still get an empty results.md (header + zero rows).
    for chip_dir in sorted(RESULTS.iterdir()):
        if not chip_dir.is_dir() or not chip_dir.name.startswith("apple-"):
            continue
        out = chip_dir / "results.md"
        out.write_text(render(chip_dir.name, s.get(chip_dir.name, {})))
        n = len(s.get(chip_dir.name, {}))
        print(f"[render] wrote {out} ({n} kernels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
