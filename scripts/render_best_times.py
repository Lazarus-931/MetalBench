#!/usr/bin/env python3
"""Regenerate best_times.md from session.json. Auto-invoked by `make refresh`."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    s = json.loads((REPO / "session.json").read_text())

    kernels: list[tuple[str, str]] = []
    for setname in ("common", "standard", "full"):
        seen: set[str] = set()
        for entry in sorted(os.listdir(REPO / "metal" / "kernels" / setname)):
            if entry in ("utils", "TEMPLATE.metal", ".gitkeep"):
                continue
            name = entry[:-6] if entry.endswith(".metal") else entry
            if name in seen:
                continue
            seen.add(name)
            kernels.append((setname, name))

    chips = ["apple-m1", "apple-m2", "apple-m3", "apple-m4", "apple-m5"]

    def cell(chip: str, name: str) -> str:
        entry = s.get(chip, {}).get(name)
        if not entry:
            return "—"
        ms = entry.get("best_time_ms")
        sp = entry.get("speedup_vs_mlx")
        if not isinstance(ms, (int, float)) or not isinstance(sp, (int, float)) or sp <= 0:
            return "—"
        return f"{ms:.3f} ({sp:.2f}×)"

    lines = [
        "# MetalBench Best Times",
        "",
        "Best kernel time + speedup vs MLX per chip. `—` = not yet benchmarked on that chip.",
        "Auto-generated from `session.json` by `scripts/render_best_times.py`. Do not hand-edit.",
        "",
        "| kernel | set | M1 | M2 | M3 | M4 | M5 |",
        "|---|---|---|---|---|---|---|",
    ]
    for setname, name in kernels:
        cells = " | ".join(cell(c, name) for c in chips)
        lines.append(f"| `{name}` | {setname} | {cells} |")
    lines += [
        "",
        f"_{len(kernels)} kernels total. "
        f"Chips covered: M2 ({len(s.get('apple-m2', {}))}), "
        f"M4 ({len(s.get('apple-m4', {}))})._",
    ]

    (REPO / "best_times.md").write_text("\n".join(lines) + "\n")
    print(f"[render-best-times] wrote best_times.md ({len(kernels)} kernels × {len(chips)} chips)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
