"""Apple M4 family chip specs."""
from __future__ import annotations

VARIANTS: dict[str, dict] = {
    "base": {
        "name": "Apple M4",
        "gpu_cores": 10,
        "peak_TFLOPS_fp32": 4.6,
        "peak_BW_GBps": 120.0,
        "simdgroup_width": 32,
        "tg_mem_max_bytes": 32 * 1024,
        "max_threads_per_tg": 1024,
    },
    "pro": {
        "name": "Apple M4 Pro",
        "gpu_cores": 20,
        "peak_TFLOPS_fp32": 9.2,
        "peak_BW_GBps": 273.0,
        "simdgroup_width": 32,
        "tg_mem_max_bytes": 32 * 1024,
        "max_threads_per_tg": 1024,
    },
    "max": {
        "name": "Apple M4 Max",
        "gpu_cores": 40,
        "peak_TFLOPS_fp32": 18.4,
        "peak_BW_GBps": 546.0,
        "simdgroup_width": 32,
        "tg_mem_max_bytes": 32 * 1024,
        "max_threads_per_tg": 1024,
    },
}


from . import m2 as _m2_synth


def derive(parsed_trace: dict, bench_timing: dict, variant: str = "base") -> dict:
    """Rough M4 synthesizer. Delegates to m2 algorithm with M4 chip constants.
    Per-pattern multipliers are not yet validated against M4 Xcode CSVs."""
    spec = dict(VARIANTS.get(variant, VARIANTS["base"]))
    out = _m2_synth.derive(
        parsed_trace, bench_timing, variant="base", spec_override=spec,
    )
    out["chip"] = spec["name"]
    out["variant"] = variant
    out["_synthesizer_note"] = (
        "Rough M4 synthesis: M2 algorithm + M4 peak/core constants. "
        "Per-pattern multipliers pending validation against M4 Xcode CSVs."
    )
    return out
