"""Wall-clock timing of MLX callables for the harness baseline."""
from __future__ import annotations
import time
from typing import Any, Callable

import mlx.core as mx


def _peak_reset():
    fn = getattr(mx, "reset_peak_memory", None) or getattr(mx.metal, "reset_peak_memory", None)
    if fn: fn()


def _peak_get():
    fn = getattr(mx, "get_peak_memory", None) or getattr(mx.metal, "get_peak_memory", None)
    return int(fn()) if fn else None


def _clear_cache():
    fn = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
    if fn: fn()


def warm_jit(fn: Callable[..., Any], *args, n: int = 1) -> None:
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
    """Time `fn(*args)`. mx.synchronize() after each call ensures the GPU
    queue has drained before stopping the clock."""
    if cold_start:
        _clear_cache()
        warmup = 0
    if track_memory:
        _peak_reset()

    for _ in range(warmup):
        out = fn(*args); mx.eval(out)
    mx.synchronize()

    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args); mx.eval(out); mx.synchronize()
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
    peak = _peak_get()
    if track_memory and peak is not None:
        result["peak_memory_bytes"] = peak
    return result


def clear_caches() -> None:
    _clear_cache()
