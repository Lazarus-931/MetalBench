"""gputrace_check — pre-compute every diagnostic field the Profiler LLM needs.

The Profiler agent's downstream LLM should never have to derive metrics from
raw bytes. This module ingests:

    * a parsed `.gputrace` bundle  (optional — captures what Metal *actually*
      dispatched: function name, grid, threadgroup, buffer bindings)
    * the kernel's registry entry  (what we *promised* would be dispatched:
      grid, threadgroup, input_shapes, flops, bytes)
    * a live `BenchResult`         (timing, throughput, occupancy from the
      most recent harness run — .gputrace has NO timing data, by design)
    * the kernel's `session.json` record (best-so-far, pso_max_threads_per_tg)

…and returns one flat-ish dict with every field already computed and named
for at-a-glance LLM consumption.

Why this exists
---------------
Apple's .gputrace is a command-intent recording, not a profile. It has no
timestamps, no counter samples, no occupancy. So this module FUSES the trace
(authoritative for "what shape did Metal actually run?") with the live bench
(authoritative for "how long did it take?") and the registry (authoritative
for "what should it have been?"). The mismatches it catches are real bugs
the harness has missed in the past — e.g. `conv_transpose2d_sub_tanh` on M2
silently capped threadgroup at 896 while we'd requested 1024, leaving the
output buffer all zeros and the bench still reporting numbers.

Example output (relu on M2, healthy)
-------------------------------------
    {
      "kernel": "relu",
      "chip": "Apple M2 (m2)  8 CPU / 8 GPU / 9 GB",
      "correct": true,
      "dispatch_check": {
        "function_dispatched_matches_registry": null,  # name not in trace
        "registry_function": "relu_f32",
        "trace_function": null,
        "grid_matches_registry": true,
        "registry_grid": [65536, 1, 1],
        "trace_grid": [65536, 1, 1],
        "threadgroup_matches_registry": true,
        "registry_threadgroup": [1024, 1, 1],
        "trace_threadgroup": [1024, 1, 1],
        "threadgroup_within_pso_limit": true,
        "tg_threads_requested": 1024,
        "pso_max_threads_per_tg": 1024,
        "total_threads_dispatched": 65536
      },
      "buffer_check": [ ... ],
      "timing": {
        "min_ms": 0.012, "median_ms": 0.022, "mean_ms": 0.039,
        "mean_to_median_ratio": 1.77,
        "is_sub_resolution": true,
        "is_thermally_jittery": true
      },
      "throughput": {"gflops": 11.8, "gbps": 94.3, "arith_intensity": 0.12},
      "roofline": {
        "classification": "memory-bound (latency-dominated <50µs)",
        "sol": 0.94, "sol_compute_pct": 0.3, "sol_memory_pct": 94.3,
        "headroom_compute_pct": 99.7, "headroom_memory_pct": 5.7,
        "dominant_headroom": "compute",
        "suggest": "kernel-launch overhead dominates — ..."
      },
      "occupancy_estimates": {
        "tg_static_mem_bytes": 0,
        "tg_static_mem_per_thread_bytes": 0,
        "note": "Real ALU/SIMD utilization is not exposed by Apple ..."
      },
      "history": {"current_best_ms": 0.017, "previous_attempts_count": 0}
    }
"""
from __future__ import annotations

import math
from typing import Any

# Roofline lives in mlx/scripts/; the existing profiler agent already imports
# it via sys.path. We mirror that approach to avoid coupling.
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "mlx" / "scripts"))
try:
    import roofline as _roofline  # type: ignore
finally:
    sys.path.pop(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shape_bytes(shape, dtype_bytes: int = 4) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n * dtype_bytes


def _product(xs) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p


def _chip_id(chip: str | None) -> str:
    if not chip:
        return "m2"
    for g in ("M5", "M4", "M3", "M2", "M1"):
        if g in chip:
            return g.lower()
    return "m2"


def _first_dispatch(parsed_trace: dict) -> dict | None:
    for cb in parsed_trace.get("command_buffers", []) or []:
        for d in cb.get("dispatches", []) or []:
            return d
    return None


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def gputrace_check(
    trace_path: str | None,
    registry_entry: dict,
    bench,                              # BenchResult (duck-typed)
    session_record: dict | None = None,
) -> dict[str, Any]:
    """Build the pre-computed diagnostic packet. See module docstring."""
    out: dict[str, Any] = {
        "kernel": getattr(bench, "kernel", None),
        "chip": getattr(bench, "chip", None),
        "correct": getattr(bench, "correct", None),
        "max_err": getattr(bench, "max_err", None),
    }

    # ------------------------------------------------------------------
    # 1. Parse trace (best-effort)
    # ------------------------------------------------------------------
    trace_dispatch: dict | None = None
    trace_parse_error: str | None = None
    if trace_path:
        try:
            from .gputrace import parse as _parse
            parsed = _parse(trace_path)
            trace_dispatch = _first_dispatch(parsed)
            out["trace_path"] = parsed.get("bundle_path")
            out["trace_n_dispatches"] = sum(
                len(cb.get("dispatches", []) or [])
                for cb in parsed.get("command_buffers", []) or []
            )
        except Exception as e:  # noqa: BLE001
            trace_parse_error = str(e)

    # ------------------------------------------------------------------
    # 2. Dispatch correctness check
    # ------------------------------------------------------------------
    reg_fn = registry_entry.get("metal_function")
    reg_grid = list(registry_entry.get("grid", []) or [])
    reg_tg = list(registry_entry.get("threadgroup", []) or [])

    pso_limit = (
        (session_record or {}).get("pso_max_threads_per_tg")
        or getattr(bench, "max_threads_per_tg", None)
    )

    if trace_dispatch is not None:
        t_fn = trace_dispatch.get("function")
        t_grid = list(trace_dispatch.get("grid", []) or [])
        t_tg = list(trace_dispatch.get("threadgroup", []) or [])
        tg_requested = _product(t_tg) if t_tg else _product(reg_tg)

        def _eq(a, b):
            return list(a) == list(b) if a and b else None

        # function name often absent in our harness's captures (resources land
        # in device-resources-* with no labels). Don't treat that as a mismatch.
        fn_match: bool | None
        if t_fn and reg_fn:
            fn_match = (t_fn == reg_fn)
        else:
            fn_match = None

        out["dispatch_check"] = {
            "function_dispatched_matches_registry": fn_match,
            "registry_function": reg_fn,
            "trace_function": t_fn,
            "grid_matches_registry": _eq(t_grid, reg_grid),
            "registry_grid": reg_grid,
            "trace_grid": t_grid,
            "threadgroup_matches_registry": _eq(t_tg, reg_tg),
            "registry_threadgroup": reg_tg,
            "trace_threadgroup": t_tg,
            "threadgroup_within_pso_limit": (
                (tg_requested <= pso_limit) if pso_limit else None
            ),
            "tg_threads_requested": tg_requested,
            "pso_max_threads_per_tg": pso_limit,
            "total_threads_dispatched": _product(t_grid) if t_grid else None,
        }
    else:
        # No trace available — still emit the registry-side info so the LLM
        # at least sees what was promised, and surface the PSO-limit check
        # (this is the silent-dispatch-failure detector).
        tg_requested = _product(reg_tg) if reg_tg else None
        out["dispatch_check"] = {
            "function_dispatched_matches_registry": None,
            "registry_function": reg_fn,
            "trace_function": None,
            "grid_matches_registry": None,
            "registry_grid": reg_grid,
            "trace_grid": None,
            "threadgroup_matches_registry": None,
            "registry_threadgroup": reg_tg,
            "trace_threadgroup": None,
            "threadgroup_within_pso_limit": (
                (tg_requested <= pso_limit)
                if (tg_requested and pso_limit) else None
            ),
            "tg_threads_requested": tg_requested,
            "pso_max_threads_per_tg": pso_limit,
            "total_threads_dispatched": _product(reg_grid) if reg_grid else None,
            "_note": (
                f"no trace ({trace_parse_error})" if trace_parse_error
                else "no trace supplied"
            ),
        }

    # ------------------------------------------------------------------
    # 3. Buffer sanity
    # ------------------------------------------------------------------
    buffer_check: list[dict] = []
    input_shapes = registry_entry.get("input_shapes") or []
    input_bindings = registry_entry.get("input_bindings") or ()
    expected_by_index: dict[int, int] = {}
    for idx, shape in zip(input_bindings, input_shapes):
        expected_by_index[int(idx)] = _shape_bytes(shape)

    if trace_dispatch is not None:
        for b in trace_dispatch.get("buffers", []) or []:
            idx = b.get("index")
            actual = b.get("length")
            exp = expected_by_index.get(int(idx)) if idx is not None else None
            matches: bool | None
            if exp is None or actual is None:
                matches = None
            else:
                matches = (int(actual) == int(exp))
            buffer_check.append({
                "binding_index": idx,
                "expected_bytes": exp,
                "actual_bytes": actual,
                "matches": matches,
                "label": b.get("label"),
            })
    out["buffer_check"] = buffer_check

    # ------------------------------------------------------------------
    # 4. Timing + thermal/sub-resolution flags
    # ------------------------------------------------------------------
    median = getattr(bench, "kernel_ms", None)
    mean = getattr(bench, "kernel_ms_mean", None)
    min_ms = getattr(bench, "kernel_ms_min", None)
    mtm = (mean / median) if (median and mean) else None
    out["timing"] = {
        "min_ms": min_ms,
        "median_ms": median,
        "mean_ms": mean,
        "mean_to_median_ratio": (round(mtm, 3) if mtm is not None else None),
        "stability": getattr(bench, "stability", None),
        "speedup_vs_mlx": getattr(bench, "speedup", None),
        "mlx_median_ms": getattr(bench, "mlx_ms", None),
        "is_sub_resolution": (median is not None and median < 0.005),
        "is_thermally_jittery": (mtm is not None and mtm > 1.5),
    }

    # ------------------------------------------------------------------
    # 5. Throughput + roofline + headroom + bottleneck label
    # ------------------------------------------------------------------
    flops = float(registry_entry.get("flops") or 0.0)
    bytes_ = float(registry_entry.get("bytes") or 0.0)
    chip_g = _chip_id(getattr(bench, "chip", None))
    rl = _roofline.classify(chip_g, flops, bytes_, median or 0.001)
    sol_c = float(rl.get("sol_compute") or 0.0)
    sol_m = float(rl.get("sol_memory") or 0.0)
    head_c = max(0.0, (1.0 - sol_c) * 100.0)
    head_m = max(0.0, (1.0 - sol_m) * 100.0)
    dominant = "compute" if head_c >= head_m else "memory"

    out["throughput"] = {
        "gflops": getattr(bench, "gflops", None),
        "gbps": getattr(bench, "gbps", None),
        "arith_intensity": getattr(bench, "arith_intensity", None),
    }
    out["roofline"] = {
        "classification": rl.get("classification"),
        "sol": rl.get("sol"),
        "sol_compute_pct": round(sol_c * 100.0, 2),
        "sol_memory_pct": round(sol_m * 100.0, 2),
        "headroom_compute_pct": round(head_c, 2),
        "headroom_memory_pct": round(head_m, 2),
        "dominant_headroom": dominant,
        "ridge_intensity": rl.get("ridge"),
        "peak_compute_TFLOPS": (rl.get("peak") or {}).get("compute_TFLOPS"),
        "peak_bandwidth_GBps": (rl.get("peak") or {}).get("bw_GBps"),
        "suggest": rl.get("suggest"),
    }
    out["bottleneck_label"] = rl.get("classification")

    # ------------------------------------------------------------------
    # 6. Occupancy / ALU estimates
    # ------------------------------------------------------------------
    tg_mem = (
        (session_record or {}).get("tg_static_mem_bytes")
        if session_record else None
    )
    if tg_mem is None:
        tg_mem = getattr(bench, "tg_mem_bytes", None)
    tg_thr = _product(reg_tg) if reg_tg else None
    mem_per_thread = (
        (tg_mem / tg_thr) if (tg_mem is not None and tg_thr) else None
    )
    out["occupancy_estimates"] = {
        "tg_static_mem_bytes": tg_mem,
        "tg_threads": tg_thr,
        "tg_static_mem_per_thread_bytes": (
            round(mem_per_thread, 3) if mem_per_thread is not None else None
        ),
        "pso_max_threads_per_tg": pso_limit,
        "tg_fill_ratio": (
            round(tg_thr / pso_limit, 3)
            if (tg_thr and pso_limit) else None
        ),
        "note": (
            "Real ALU/SIMD utilization and computeKernelInvocations are not "
            "exposed by Apple's public Metal counter sample sets on consumer "
            "M-series silicon (MTLCommonCounterSetStatistic returns no usable "
            "data outside of Apple's internal tools). We surface only what we "
            "CAN measure: threadgroup static memory pressure and the fill "
            "ratio of the threadgroup vs the PSO's max. Genuine occupancy "
            "must be inferred from achieved GB/s vs roofline."
        ),
    }

    # ------------------------------------------------------------------
    # 7. History context
    # ------------------------------------------------------------------
    if session_record:
        out["history"] = {
            "current_best_ms": session_record.get("best_time_ms"),
            "current_best_gflops": session_record.get("gflops"),
            "current_best_gbps": session_record.get("gbps"),
            "current_best_stability": session_record.get("stability"),
            "previous_attempts_count": session_record.get("iters"),
            "updated_at": session_record.get("updated_at"),
        }
    else:
        out["history"] = None

    return out
