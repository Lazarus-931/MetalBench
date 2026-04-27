from __future__ import annotations
import time
from typing import Any, Callable

import numpy as np
import mlx.core as mx

from .task import Task
from .host import run_kernel


def _time_mlx(fn: Callable[..., Any], *args, warmup: int, iters: int) -> tuple[Any, dict]:
    """Time an MLX callable. mx.eval forces materialization (lazy graph)."""
    out = None
    for _ in range(warmup):
        out = fn(*args)
        mx.eval(out)
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return out, {
        "min_ms":    times[0],
        "median_ms": times[len(times) // 2],
        "mean_ms":   sum(times) / len(times),
        "iters":     iters,
    }


def _allclose(actual: mx.array, expected: mx.array, rtol: float, atol: float):
    a = np.asarray(actual)
    b = np.asarray(expected)
    if a.shape != tuple(b.shape):
        return False, float("inf")
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    tol  = atol + rtol * np.abs(b.astype(np.float64))
    max_err = float(diff.max()) if a.size else 0.0
    return bool(np.all(diff <= tol)), max_err


def evaluate(task: Task, *, seed: int = 0, warmup: int = 5, iters: int = 50) -> dict:
    """Run one task: dispatch the kernel, run the MLX reference, compare + time."""
    inputs = list(task.make_inputs(seed))
    for x in inputs:
        mx.eval(x)

    kernel_outs, k_t = run_kernel(task, inputs, warmup=warmup, iters=iters)

    ref = task.reference(*inputs)
    ref_outs = list(ref) if isinstance(ref, (list, tuple)) else [ref]
    for r in ref_outs:
        mx.eval(r)
    _, r_t = _time_mlx(task.reference, *inputs, warmup=warmup, iters=iters)

    if len(kernel_outs) != len(ref_outs):
        raise RuntimeError(
            f"task {task.name}: kernel produced {len(kernel_outs)} outputs, "
            f"reference produced {len(ref_outs)}"
        )

    per_output = []
    for k, r in zip(kernel_outs, ref_outs):
        ok, max_err = _allclose(k, r, task.rtol, task.atol)
        per_output.append({"ok": ok, "max_err": max_err})

    correct = all(o["ok"] for o in per_output)
    speedup = (r_t["median_ms"] / k_t["median_ms"]) if k_t["median_ms"] > 0 else float("inf")

    return {
        "task":             task.name,
        "correct":          correct,
        "speedup":          speedup,
        "kernel_timing":    k_t,
        "reference_timing": r_t,
        "outputs":          per_output,
    }
