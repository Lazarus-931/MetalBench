"""Roofline classification for MetalBench kernels.

Given (flops, bytes, median_ms, chip_type), classifies the kernel as
compute-bound / memory-bound / balanced and reports speed-of-light fraction.
Suggests an optimization direction.

Peaks below are approximate base-chip numbers. Pro/Max/Ultra variants of the
same generation scale linearly in compute and bandwidth, so the ratio (and
thus the ridge point) is roughly preserved.
"""

# Derived from chips.json via agent_steel.chips — single source of truth.
def _load_from_registry():
    import sys as _sys
    from pathlib import Path as _Path
    _REPO = _Path(__file__).resolve().parents[2]
    if str(_REPO) not in _sys.path:
        _sys.path.insert(0, str(_REPO))
    from agent_steel import chips as _chips
    peaks = {c.gen: dict(bw_GBps=c.peak_bandwidth_GBps,
                         compute_TFLOPS=c.peak_compute_TFLOPS)
             for c in _chips.CHIPS}
    # oldest -> newest is the original iteration order in _generation()
    gens = tuple(reversed(_chips.list_generations()))
    return peaks, gens, _chips.DEFAULT_FALLBACK_GEN

CHIP_PEAKS, _GEN_ORDER, _FALLBACK_GEN = _load_from_registry()


def _generation(chip_type: str) -> str:
    for g in _GEN_ORDER:
        if chip_type.startswith(g):
            return g
    return _FALLBACK_GEN


def classify(chip_type: str, flops: float, bytes_: float, median_ms: float):
    peak = CHIP_PEAKS[_generation(chip_type)]
    intensity = (flops / bytes_) if bytes_ > 0 else 0.0
    ridge = peak["compute_TFLOPS"] * 1000.0 / peak["bw_GBps"]

    seconds = median_ms / 1000.0
    gflops = (flops / seconds / 1e9) if seconds > 0 else 0.0
    gbps   = (bytes_ / seconds / 1e9) if seconds > 0 else 0.0

    sol_compute = gflops / (peak["compute_TFLOPS"] * 1000.0)
    sol_memory  = gbps / peak["bw_GBps"]

    if flops == 0:
        cls, sol = "memory-bound", sol_memory
    elif intensity > ridge:
        cls, sol = "compute-bound", sol_compute
    elif intensity > ridge * 0.5:
        cls, sol = "balanced",      max(sol_compute, sol_memory)
    else:
        cls, sol = "memory-bound",  sol_memory

    if median_ms < 0.05:
        cls = cls + " (latency-dominated <50µs)"
        sol = max(sol, 0.0)

    if cls.startswith("memory"):
        suggest = "float4 grid-stride, larger grid, fewer device reads, threadgroup-mem caching"
    elif cls.startswith("compute"):
        suggest = "simdgroup_matrix MMA tiles, inner-loop unroll, FMA fusion, double-buffer"
    else:
        suggest = "double-buffer loads, larger tile, async copies — already near both ceilings"
    if "latency" in cls:
        suggest = "kernel-launch overhead dominates — bigger work-per-launch, batch up calls"

    return dict(
        intensity=intensity,
        ridge=ridge,
        gflops=gflops,
        gbps=gbps,
        sol=sol,
        sol_compute=sol_compute,
        sol_memory=sol_memory,
        classification=cls,
        suggest=suggest,
        peak=peak,
    )


def format_line(metrics) -> str:
    return (
        f"  roofline    : {metrics['classification']}  "
        f"sol={metrics['sol']*100:.0f}%  "
        f"intensity={metrics['intensity']:.2f} vs ridge {metrics['ridge']:.1f}\n"
        f"              {metrics['suggest']}"
    )
