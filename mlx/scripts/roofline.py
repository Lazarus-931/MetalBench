"""Roofline classification for MetalBench kernels.

Given (flops, bytes, median_ms, chip_type), classifies the kernel as
compute-bound / memory-bound / balanced and reports speed-of-light fraction.
Suggests an optimization direction.

Peaks below are approximate base-chip numbers. Pro/Max/Ultra variants of the
same generation scale linearly in compute and bandwidth, so the ratio (and
thus the ridge point) is roughly preserved.
"""

CHIP_PEAKS = {
    "m1": dict(bw_GBps=68.25,  compute_TFLOPS=2.6),
    "m2": dict(bw_GBps=100.0,  compute_TFLOPS=3.6),
    "m3": dict(bw_GBps=102.4,  compute_TFLOPS=4.1),
    "m4": dict(bw_GBps=120.0,  compute_TFLOPS=4.5),
    "m5": dict(bw_GBps=150.0,  compute_TFLOPS=5.5),
}


def _generation(chip_type: str) -> str:
    for g in ("m1", "m2", "m3", "m4", "m5"):
        if chip_type.startswith(g):
            return g
    return "m2"


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
