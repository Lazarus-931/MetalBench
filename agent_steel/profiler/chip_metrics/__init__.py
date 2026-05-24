"""Chip-aware metrics derivation.

Each M-series generation lives in its own module (m1.py, m2.py, ...). A module
must expose a `VARIANTS: dict[str, dict]` mapping variant name ("base", "pro",
"max", "ultra") to a spec dict with at least:

    name              : str   e.g. "Apple M2 Pro"
    gpu_cores         : int
    peak_TFLOPS_fp32  : float
    peak_BW_GBps      : float
    simdgroup_width   : int   (always 32 for Apple GPUs but kept per-chip for safety)
    tg_mem_max_bytes  : int
    max_threads_per_tg: int

Adding a new generation = drop in `m6.py` with a VARIANTS dict. No edits here.
"""
from __future__ import annotations
import importlib
import pkgutil
from typing import Any


def _load(generation: str) -> Any:
    return importlib.import_module(f".{generation}", package=__name__)


def list_generations() -> list[str]:
    return [
        name for _, name, _ in pkgutil.iter_modules(__path__)
        if name.startswith("m") and name[1:].isdigit()
    ]


def get_chip_specs(generation: str, variant: str = "base") -> dict:
    """Return the spec dict for (generation, variant). Raises if unknown."""
    g = generation.lower().lstrip("apple-").lstrip("apple_")
    mod = _load(g)
    variants = getattr(mod, "VARIANTS", {})
    if variant not in variants:
        raise KeyError(
            f"variant {variant!r} not in {g}.VARIANTS "
            f"(available: {list(variants)})"
        )
    return variants[variant]


def derive_metrics(
    *,
    bench: dict,
    parsed_trace: dict | None,
    generation: str,
    variant: str = "base",
    xcode_csv: str | None = None,
) -> dict:
    """Turn raw bench + parsed gputrace into chip-aware utilization metrics.

    Returns a dict with:
      chip, sol_compute_pct, sol_bw_pct, arithmetic_intensity,
      avg_ALU_utilization (proxy), notes (list of strings).
    """
    spec = get_chip_specs(generation, variant)
    kernel_ms = bench.get("kernel_ms")
    flops = bench.get("flops")
    bytes_moved = bench.get("bytes")

    out: dict[str, Any] = {
        "chip": spec.get("name", generation),
        "variant": variant,
        "spec_used": {
            "peak_TFLOPS_fp32": spec["peak_TFLOPS_fp32"],
            "peak_BW_GBps": spec["peak_BW_GBps"],
            "gpu_cores": spec["gpu_cores"],
        },
        "notes": [],
    }

    if kernel_ms and kernel_ms > 0 and flops:
        achieved_tflops = (flops / 1e12) / (kernel_ms / 1000.0)
        out["achieved_TFLOPS"] = achieved_tflops
        out["sol_compute_pct"] = 100.0 * achieved_tflops / spec["peak_TFLOPS_fp32"]
    if kernel_ms and kernel_ms > 0 and bytes_moved:
        achieved_bw = (bytes_moved / 1e9) / (kernel_ms / 1000.0)
        out["achieved_BW_GBps"] = achieved_bw
        out["sol_bw_pct"] = 100.0 * achieved_bw / spec["peak_BW_GBps"]
    if flops and bytes_moved:
        out["arithmetic_intensity"] = flops / bytes_moved

    # ALU utilization is a proxy: there is no public counter, but
    # achieved_TFLOPS / peak is the best low-bias estimate.
    if "sol_compute_pct" in out:
        out["avg_ALU_utilization"] = out["sol_compute_pct"] / 100.0
        out["notes"].append(
            "avg_ALU_utilization is a throughput-derived proxy "
            "(achieved FLOPs / peak FLOPs). Apple does not expose true "
            "ALU-active counters in .gputrace."
        )

    if parsed_trace and parsed_trace.get("dispatches"):
        d0 = parsed_trace["dispatches"][0]
        grid = d0.get("grid")
        tg = d0.get("threadgroup")
        if grid and tg:
            total_threads = grid[0] * grid[1] * grid[2]
            tg_threads = tg[0] * tg[1] * tg[2]
            simdgroups_per_tg = (tg_threads + spec["simdgroup_width"] - 1) // spec["simdgroup_width"]
            out["dispatch_geometry"] = {
                "total_threads": total_threads,
                "tg_threads": tg_threads,
                "simdgroups_per_tg": simdgroups_per_tg,
                "tg_mem_headroom_bytes": spec["tg_mem_max_bytes"],
            }

    # NOTE: Xcode CSV ingestion has been removed from the runtime synthesizer.
    # The synthesizer must work from `.gputrace` + bench timing + chip-spec
    # constants alone. CSV reading lives only in scripts/validate_synthesizer.py
    # for offline oracle comparison.
    if xcode_csv:
        raise RuntimeError(
            "derive_metrics(): xcode_csv argument is no longer accepted. "
            "Use scripts/validate_synthesizer.py for offline CSV comparison."
        )

    # Call into the active per-chip synthesizer if available.
    mod = _load(generation.lower().lstrip("apple-").lstrip("apple_"))
    if hasattr(mod, "derive"):
        synth = mod.derive(parsed_trace or {}, bench, variant=variant)
        out["synth"] = synth

    return out
