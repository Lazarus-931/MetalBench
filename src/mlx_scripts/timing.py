"""MLX timing utilities.

Wall-clock timing of MLX callables, with explicit GPU synchronization and
peak-memory tracking. Used by the harness to measure the baseline against
which a Metal kernel is compared.

Why `mx.synchronize()` and not just `mx.eval()`:
    `mx.eval(out)` materializes the lazy graph and blocks for the result, but
    using `mx.synchronize()` after is the documented way to guarantee the
    GPU command queue has drained before stopping the clock. Without it,
    cheap kernels can return artificially low numbers on noisy systems.
"""
from __future__ import annotations
from typing import Any, Callable

import mlx.core as mx


def warm_jit(fn: Callable[..., Any], *args, n: int = 1) -> None:
    """Trigger graph compilation + kernel cache before the warmup loop.

    Separated from warmup so the very first iteration's JIT cost doesn't
    leak into warmup statistics if you ever want to record those.
    """
    for _ in range(n):
        out = fn(*args)
        mx.eval(out)
    mx.synchronize()


def time_mlx(
    fn: Callable[..., Any],
    *args,
    warmup: int = 5,
    iters: int = 50,
    track_memory: bool = True,
    cold_start: bool = False,
) -> dict:
    """Time an MLX callable.

    Returns:
        {
          "min_ms", "median_ms", "mean_ms", "iters",
          "peak_memory_bytes" (if track_memory),
          "cold_start" (if cold_start),
        }

    Args:
        cold_start: clear the Metal kernel + buffer cache before the warmup,
            then *do not* warm up. Useful when you specifically want to
            measure first-launch latency. Default False.
        track_memory: reset peak GPU memory before the timed loop and report
            it after. Cheap to leave on.
    """
    if cold_start:
        if hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
        warmup_eff = 0
    else:
        warmup_eff = warmup

    if track_memory and hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()

    out = None
    for _ in range(warmup_eff):
        out = fn(*args)
        mx.eval(out)
    mx.synchronize()

    import time
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    result: dict = {
        "min_ms":    times[0],
        "median_ms": times[len(times) // 2],
        "mean_ms":   sum(times) / len(times),
        "iters":     iters,
    }
    if cold_start:
        result["cold_start"] = True
    if track_memory and hasattr(mx.metal, "get_peak_memory"):
        result["peak_memory_bytes"] = int(mx.metal.get_peak_memory())
    return result


def memory_snapshot() -> dict:
    """Current MLX/Metal allocator state. For one-off inspection."""
    snap: dict = {}
    for attr in ("get_active_memory", "get_peak_memory", "get_cache_memory"):
        fn = getattr(mx.metal, attr, None)
        if fn:
            try:
                snap[attr.removeprefix("get_")] = int(fn())
            except Exception:
                pass
    return snap


def clear_caches() -> None:
    """Drop MLX's compiled-kernel + buffer caches.

    Call between unrelated benchmarks if you want each one to pay its own
    JIT/allocation cost. Within a single benchmark's warmup+iters loop you
    do *not* want this — you'd be timing recompilation.
    """
    if hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()
