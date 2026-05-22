#!/usr/bin/env python3
"""Regenerate results/<chip>/results.md from per-kernel JSON files.

The website (Lazarus-931.github.io/leaderboard.html) fetches
`results/<chip>/results.md` over raw.githubusercontent.com, so this file must
stay current after agent-driven kernel updates.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def render(chip_dir: Path) -> str:
    rows = []
    for jf in sorted(chip_dir.glob("*.json")):
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        if d.get("task") is None:
            continue
        m = d.get("metrics", {}) or {}
        speedup = d.get("speedup") or m.get("speedup_vs_mlx")
        ms = m.get("metal_ms_min") or m.get("metal_ms") or d.get("kernel_timing", {}).get("min_ms")
        gflops = m.get("gflops")
        gbps = m.get("gbps") or m.get("bandwidth_gbps")
        if ms is None:
            continue
        rows.append({
            "name": d["task"],
            "ms": float(ms),
            "speedup": float(speedup) if speedup else 0.0,
            "gflops": float(gflops) if gflops else 0.0,
            "gbps": float(gbps) if gbps else 0.0,
        })
    rows.sort(key=lambda r: r["name"])
    out = [f"# {chip_dir.name} Results", ""]
    out += ["| kernel | time (ms) | speedup | GFLOPS | GB/s |", "|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['name']} | {r['ms']:.3f} | {r['speedup']:.2f}× | {int(r['gflops'])} | {int(r['gbps'])} |"
        )
    out += ["", f"_{len(rows)} kernels._", ""]
    return "\n".join(out) + "\n"


def main() -> int:
    if not RESULTS.exists():
        sys.exit(f"missing {RESULTS}")
    for chip_dir in sorted(RESULTS.iterdir()):
        if not chip_dir.is_dir() or not chip_dir.name.startswith("apple-"):
            continue
        out = chip_dir / "results.md"
        out.write_text(render(chip_dir))
        print(f"[render] wrote {out} ({sum(1 for _ in chip_dir.glob('*.json'))} kernels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
