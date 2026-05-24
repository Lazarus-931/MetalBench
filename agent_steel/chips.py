"""Single source of truth for Apple M-series chip metadata.

Reads `chips.json` at the repository root and exposes:

- `CHIPS`               — list of `Chip` dataclasses (newest -> oldest order).
- `detect_generation()` — map a brand string ("Apple M2 Max") -> generation tag ("m2").
- `list_generations()`  — ordered tuple of generation tags ("m5", "m4", ...).
- `list_chip_types()`   — flat (needle, tag) list matching the old `_CHIP_TYPES` tuple.
- `peak()`              — peak {bw_GBps, compute_TFLOPS} for a generation.
- `ceiling()`           — {tg_memory_max_bytes, peak_compute_TFLOPS, peak_bandwidth_GBps}.

Adding a new generation (e.g. M6) is a one-file edit to `chips.json` — every
consumer in MetalBench picks it up automatically at next import / next `make`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "chips.json"


@dataclass(frozen=True)
class Chip:
    gen: str                       # "m2"
    brand: str                     # "M2"
    variants: Tuple[str, ...]      # ("Ultra", "Max", "Pro", "")
    peak_compute_TFLOPS: float
    peak_bandwidth_GBps: float
    tg_memory_max_bytes: int

    def variant_tags(self) -> Tuple[Tuple[str, str], ...]:
        """Return ((needle, tag), ...) for this chip, ordered most-specific first.

        Example for M2: (("M2 Ultra","m2_ultra"), ("M2 Max","m2_max"),
                         ("M2 Pro","m2_pro"), ("M2","m2"))
        """
        out = []
        for v in self.variants:
            if v == "":
                out.append((self.brand, self.gen))
            else:
                out.append((f"{self.brand} {v}", f"{self.gen}_{v.lower()}"))
        return tuple(out)


def _load() -> dict:
    with _REGISTRY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


_RAW = _load()

CHIPS: Tuple[Chip, ...] = tuple(
    Chip(
        gen=c["gen"],
        brand=c["brand"],
        variants=tuple(c["variants"]),
        peak_compute_TFLOPS=float(c["peak_compute_TFLOPS"]),
        peak_bandwidth_GBps=float(c["peak_bandwidth_GBps"]),
        tg_memory_max_bytes=int(c["tg_memory_max_bytes"]),
    )
    for c in _RAW["chips"]
)

DEFAULT_FALLBACK_GEN: str = _RAW.get("default_fallback_gen", "m2")


def list_generations() -> Tuple[str, ...]:
    """Newest -> oldest tuple of generation tags."""
    return tuple(c.gen for c in CHIPS)


def list_chip_types() -> Tuple[Tuple[str, str], ...]:
    """Flat (brand_needle, tag) list, most-specific suffix first within each
    generation, newest generation first. Matches the legacy `_CHIP_TYPES` tuple
    in `mlx/scripts/mlx_helpers.py`.
    """
    out: list[Tuple[str, str]] = []
    for c in CHIPS:
        out.extend(c.variant_tags())
    return tuple(out)


def detect_generation(brand_string: str | None, fallback: str | None = None) -> str:
    """Return the generation tag ("m2", "m5", ...) for a brand string.

    Case-insensitive substring match against `Chip.brand`. Returns the first
    match in newest-first order; falls back to `DEFAULT_FALLBACK_GEN` (or the
    caller-supplied `fallback`) if no chip matches.
    """
    if not brand_string:
        return fallback or DEFAULT_FALLBACK_GEN
    s = brand_string.upper()
    for c in CHIPS:
        if c.brand.upper() in s:
            return c.gen
    return fallback or DEFAULT_FALLBACK_GEN


def detect_variant(brand_string: str | None) -> str:
    """Return the most-specific variant tag ("m2_max", "m5", ...) for a brand
    string, or "unknown" if no chip matches. Mirrors C++ `parse_type()`."""
    if not brand_string:
        return "unknown"
    for needle, tag in list_chip_types():
        if needle in brand_string:
            return tag
    return "unknown"


def peak(gen: str) -> dict:
    """Peak compute/bandwidth dict for `gen`. Mirrors roofline.CHIP_PEAKS."""
    for c in CHIPS:
        if c.gen == gen:
            return {
                "bw_GBps": c.peak_bandwidth_GBps,
                "compute_TFLOPS": c.peak_compute_TFLOPS,
            }
    raise KeyError(f"unknown chip generation: {gen!r}")


def ceiling(gen: str) -> dict:
    """Chip-ceiling dict for `gen`. Mirrors gputrace_check._CHIP_CEILINGS."""
    for c in CHIPS:
        if c.gen == gen:
            return {
                "tg_memory_max_bytes": c.tg_memory_max_bytes,
                "peak_compute_TFLOPS": c.peak_compute_TFLOPS,
                "peak_bandwidth_GBps": c.peak_bandwidth_GBps,
            }
    raise KeyError(f"unknown chip generation: {gen!r}")


__all__ = [
    "Chip",
    "CHIPS",
    "DEFAULT_FALLBACK_GEN",
    "list_generations",
    "list_chip_types",
    "detect_generation",
    "detect_variant",
    "peak",
    "ceiling",
]
