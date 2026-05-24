# Apple M4 family chip specs for Agent Steel chip-aware metrics.
from __future__ import annotations

# Notes on M4 vs M3:
#   - GPU family is still Apple9 (Metal3), so simdgroup_width (32),
#     tg_mem_max_bytes (32 KiB) and max_threads_per_tg (1024) are unchanged
#     vs M3. See Apple Metal Feature Set Tables:
#     https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
#   - M4 doubles ray-tracing throughput vs M3 and adds 2nd-gen "Dynamic Caching"
#     (https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/),
#     but those do not change the kernel-visible limits used here.
#   - No M4 Ultra has shipped as of 2026-05; only base/pro/max are included.
#     https://en.wikipedia.org/wiki/Apple_M4

# Peak FP32 TFLOPS are derived from public per-core benchmark reporting
# (cpu-monkey / nanoreview / notebookcheck). Apple does not publish a TFLOPS
# number directly; figures used scale ~roughly 0.46 TFLOPS / GPU core, which
# matches the Apple9 generation (1.398 GHz boost * 128 ALUs * 2 FMA / core).

VARIANTS: dict[str, dict] = {
    "base": {
        # 10-core GPU variant (iMac / Mac mini / MacBook Pro 14" base / iPad Pro)
        "name": "Apple M4",
        "gpu_cores": 10,                  # https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/
        "peak_TFLOPS_fp32": 4.6,          # https://www.cpu-monkey.com/en/igpu-apple_m4_10_core
        "peak_BW_GBps": 120.0,            # https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/
        "simdgroup_width": 32,            # https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
        "tg_mem_max_bytes": 32 * 1024,    # https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
        "max_threads_per_tg": 1024,       # https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
    },
    "pro": {
        # Top-bin 20-core GPU (binned 16-core part also exists)
        "name": "Apple M4 Pro",
        "gpu_cores": 20,                  # https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/
        "peak_TFLOPS_fp32": 9.2,          # https://www.cpu-monkey.com/en/igpu-apple_m4_pro_20_core
        "peak_BW_GBps": 273.0,            # https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/
        "simdgroup_width": 32,            # https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
        "tg_mem_max_bytes": 32 * 1024,    # https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
        "max_threads_per_tg": 1024,       # https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
    },
    "max": {
        # Top-bin 40-core GPU (binned 32-core part: 410 GB/s, ~14.7 TFLOPS)
        "name": "Apple M4 Max",
        "gpu_cores": 40,                  # https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/
        "peak_TFLOPS_fp32": 18.4,         # https://www.cpu-monkey.com/en/igpu-apple_m4_max_40_core
        "peak_BW_GBps": 546.0,            # https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/
        "simdgroup_width": 32,            # https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
        "tg_mem_max_bytes": 32 * 1024,    # https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
        "max_threads_per_tg": 1024,       # https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
    },
    # "ultra": intentionally omitted — no M4 Ultra has shipped as of 2026-05.
    # https://en.wikipedia.org/wiki/Apple_M4
}


# ---------------------------------------------------------------------------
# Active synthesizer — rough port of m2.derive() with M4 chip constants.
#
# Apple M4's GPU is still Apple9 family (same as M3), sharing simdgroup width,
# tg-memory cap, and max-threads-per-tg with the Apple8 (M2) family. The
# per-pattern duty-factor *behaviour* is therefore expected to be structurally
# similar — only the peak compute/bandwidth/cores differ. We delegate to the
# CSV-validated m2 synthesizer with M4's VARIANTS as the spec_override.
#
# This is rough: it has NOT been validated against real M4 Xcode CSVs yet.
# When M4 captures land, expect to tune the per-pattern multipliers
# (alu_util_k, dram_active_frac, f32_mul, ...) inside an m4-specific path.
# ---------------------------------------------------------------------------

from . import m2 as _m2_synth


def derive(parsed_trace: dict, bench_timing: dict, variant: str = "base") -> dict:
    """Rough M4 synthesizer. Calls m2 algorithm with M4 chip constants."""
    spec = dict(VARIANTS.get(variant, VARIANTS["base"]))
    out = _m2_synth.derive(
        parsed_trace, bench_timing, variant="base", spec_override=spec,
    )
    # Re-stamp identity fields so downstream sees the right chip / variant.
    out["chip"] = spec["name"]
    out["variant"] = variant
    out["_synthesizer_note"] = (
        "Rough M4 synthesis: M2 algorithm + M4 peak/core constants. "
        "Per-pattern multipliers have not been validated against M4 Xcode CSVs yet — "
        "expect tuning when M4 captures land."
    )
    return out
