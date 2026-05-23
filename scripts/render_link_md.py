#!/usr/bin/env python3
"""Regenerate LINK.md from session.json + disk layout. Auto-invoked by `make refresh`."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/Lazarus-931/MetalBench/blob/main"


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

    def link_for(setname: str, name: str, chip_gen: str) -> str | None:
        d_dir = REPO / "metal" / "kernels" / setname / name
        d_flat = REPO / "metal" / "kernels" / setname / f"{name}.metal"
        if d_dir.is_dir():
            for cand in (d_dir / f"{chip_gen}.metal", d_dir / "default.metal"):
                if cand.is_file():
                    return str(cand.relative_to(REPO))
            return None
        if d_flat.is_file():
            return str(d_flat.relative_to(REPO))
        return None

    chips = [
        ("apple-m1", "m1"),
        ("apple-m2", "m2"),
        ("apple-m3", "m3"),
        ("apple-m4", "m4"),
        ("apple-m5", "m5"),
    ]

    lines = [
        "# LINK.md",
        "",
        f"Per-chip kernel source links — {len(kernels)} kernels × {len(chips)} chips.",
        "Chips without benchmark data show kernels as plain text (no link).",
        "",
        "Auto-generated from `session.json` + disk layout by `scripts/render_link_md.py`. Do not hand-edit.",
        "",
    ]
    for chip_id, chip_gen in chips:
        has_data = chip_id in s and len(s[chip_id]) > 0
        lines.append(f"## {chip_id}" + ("" if has_data else " _(no data)_"))
        lines.append("")
        for setname, name in kernels:
            path = link_for(setname, name, chip_gen) if has_data else None
            lines.append(f"- [{name}]({REPO_URL}/{path})" if path else f"- `{name}`")
        lines.append("")

    (REPO / "LINK.md").write_text("\n".join(lines))
    print(f"[render-link-md] wrote LINK.md ({len(kernels)} kernels × {len(chips)} chips)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
