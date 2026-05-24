#!/usr/bin/env python3
"""Generate metal/scripts/chip_table.h from chips.json.

This is the one-and-only C++-side view of the chip registry. The Makefile
invokes this before building the host binary, so adding a new chip generation
in chips.json automatically updates the C++ enum, type_name() switch, and
parse_type() ladder — no manual edits to setup.cpp/setup.h.

The generated header defines an X-macro:

    METALBENCH_CHIP_LIST(X) — expands X(EnumTag, "name", "brand needle") per
        variant in registry order (most-specific suffix first, newest gen first).

setup.h / setup.cpp consume this macro to build the enum, the type_name()
switch, and the parse_type() ladder, so the only place a new chip lives is
chips.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    src = REPO / "chips.json"
    dst = REPO / "metal" / "scripts" / "chip_table.h"
    raw = json.loads(src.read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append("// AUTO-GENERATED from chips.json by scripts/generate_chip_table.py")
    lines.append("// DO NOT EDIT BY HAND. Edit chips.json and re-run `make` (or this script).")
    lines.append("#pragma once")
    lines.append("")
    lines.append("// X-macro: X(EnumTag, \"tag_name\", \"brand needle\")")
    lines.append("// Order is most-specific-first within a generation, newest-gen-first across")
    lines.append("// generations — parse_type() relies on this ordering for correct matching.")
    lines.append("#define METALBENCH_CHIP_LIST(X) \\")

    rows: list[tuple[str, str, str]] = []
    for chip in raw["chips"]:
        brand = chip["brand"]          # "M2"
        gen = chip["gen"]              # "m2"
        for v in chip["variants"]:
            if v == "":
                enum_tag = brand.upper()
                name = gen
                needle = brand
            else:
                enum_tag = f"{brand.upper()}_{v.upper()}"
                name = f"{gen}_{v.lower()}"
                needle = f"{brand} {v}"
            rows.append((enum_tag, name, needle))

    for i, (enum_tag, name, needle) in enumerate(rows):
        sep = " \\" if i < len(rows) - 1 else ""
        lines.append(f'    X({enum_tag}, "{name}", "{needle}"){sep}')

    lines.append("")
    out = "\n".join(lines) + "\n"

    if dst.exists() and dst.read_text(encoding="utf-8") == out:
        return 0  # no-op rebuild
    dst.write_text(out, encoding="utf-8")
    print(f"generate_chip_table: wrote {dst.relative_to(REPO)} ({len(rows)} variants)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
