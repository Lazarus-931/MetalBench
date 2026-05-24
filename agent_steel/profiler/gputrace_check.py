"""gputrace_check — pre-compute every diagnostic field the Profiler LLM needs.

The Profiler agent's downstream LLM should never have to derive metrics from
raw bytes. This module ingests:

    * a parsed `.gputrace` bundle  (optional — captures what Metal *actually*
      dispatched: function name, grid, threadgroup, buffer bindings)
    * the kernel's registry entry  (what we *promised* would be dispatched:
      grid, threadgroup, input_shapes, flops, bytes)
    * a live `BenchResult`         (timing, throughput, occupancy from the
      most recent harness run — .gputrace has NO timing data, by design)
    * the .metal source text       (optional — drives the static source-analysis
      block; without it the LLM has to re-read the file from scratch)
    * the kernel's `session.json` record (best-so-far, pso_max_threads_per_tg)

…and returns one flat-ish dict with every field already computed and named
for at-a-glance LLM consumption.

v2 vs v1
--------
The first version of this packet got a 4/10 in a blind dead-test:
- `roofline.classification = "balanced"` at SOL=2% was actively misleading.
- `roofline.suggest = "near both ceilings"` at SOL=2% was actively wrong.
- `timing.*` and `throughput.*` blocks duplicated `BenchResult` verbatim.
- `occupancy_estimates.note` was a 200-word disclaimer the LLM would parrot.
- The packet didn't surface what the .metal source actually *does* —
  loop depth, simdgroup usage, vector intrinsics — exactly what an LLM
  needs to reason about a non-pattern-matchable kernel.

v2 fixes:
1. Roofline sanity gate: SOL < 5% with neither ceiling saturated → override
   classification to `under_roofline_likely_latency_or_stall` and drop the
   misleading `suggest` string.
2. New `source_analysis` block: cheap regex over the .metal — loop counts,
   simdgroup_matrix / float4 / threadgroup_mem / atomic / unroll-pragma
   presence flags. Drives every code-shape suggestion the LLM might make.
3. New `chip_ceilings` block: tg_memory_max_bytes, simdgroup_width,
   peak compute/bandwidth. Grounds tile-size and structural recommendations.
4. Dropped `timing.*` / `throughput.*` duplicates. Kept only the *flags* the
   LLM needs to gate its confidence (`is_sub_resolution`,
   `is_thermally_jittery`) under `timing_trust`.
5. `buffer_check` returns null when all entries are null (harness captures
   don't label buffers — emitting noise was misleading).
6. `occupancy_estimates.note` trimmed to a one-liner.

Apple's .gputrace is a command-intent recording, not a profile. It has no
timestamps, no counter samples, no occupancy. So this module FUSES the trace
(authoritative for "what shape did Metal actually run?") with the live bench
(authoritative for "how long did it take?"), the registry (authoritative
for "what should it have been?"), and the source (authoritative for "what
does the code actually do?"). The mismatches it catches are real bugs the
harness has missed in the past — e.g. `conv_transpose2d_sub_tanh` on M2
silently capped threadgroup at 896 while we'd requested 1024, leaving the
output buffer all zeros and the bench still reporting numbers.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

# Roofline lives in mlx/scripts/; the existing profiler agent already imports
# it via sys.path. We mirror that approach to avoid coupling.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "mlx" / "scripts"))
try:
    import roofline as _roofline  # type: ignore
finally:
    sys.path.pop(0)


# ---------------------------------------------------------------------------
# Chip ceilings — absolute budgets the LLM can use to size tiles.
# ---------------------------------------------------------------------------

# tg_memory_max_bytes is a conservative 32 KB cap that holds across
# M1-M5 Apple GPUs. peak compute / bandwidth come from roofline.CHIP_PEAKS.
# Derived from chips.json — single source of truth.
from agent_steel.chips import CHIPS as _CHIPS, ceiling as _ceiling
_CHIP_CEILINGS = {c.gen: _ceiling(c.gen) for c in _CHIPS}

# Features universal across M-family.
_COMMON_CHIP_FEATURES = {
    "simdgroup_width": 32,
    "simdgroup_matrix_supported": True,
    "unified_memory": True,
    "max_threads_per_tg_typical": 1024,
}


def _chip_ceilings_for(chip_gen: str) -> dict[str, Any]:
    ceil = _CHIP_CEILINGS.get(chip_gen) or _CHIP_CEILINGS["m2"]
    return {**ceil, **_COMMON_CHIP_FEATURES, "chip_generation": chip_gen}


# ---------------------------------------------------------------------------
# Source analysis — cheap regex over the .metal text.
# ---------------------------------------------------------------------------

def _strip_comments(s: str) -> str:
    s = re.sub(r"//[^\n]*", "", s)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    return s


_RX_KERNEL_FN = re.compile(r"\bkernel\s+\w+\s+(\w+)\s*\(", re.M)
_RX_FOR_WHILE = re.compile(r"\b(for|while)\s*\(")
_RX_THREADGROUP_DECL = re.compile(r"\bthreadgroup\s+(?:const\s+)?[\w<>:,\s]+\s+\w+\s*(?:\[|;)")
_RX_SIMDGROUP_MATRIX = re.compile(r"\bsimdgroup_matrix\b")
_RX_SIMDGROUP_LDST = re.compile(r"\bsimdgroup_(load|store|multiply_accumulate|multiply)\b")
_RX_FLOAT4 = re.compile(r"\bfloat4\b")
_RX_HALF4 = re.compile(r"\bhalf4\b")
_RX_ATOMIC = re.compile(r"\b(atomic_\w+|__metal_atomic_\w+)\b")
_RX_UNROLL = re.compile(r"#pragma\s+(?:clang\s+loop\s+)?unroll", re.I)
_RX_BARRIER = re.compile(r"\bthreadgroup_barrier\s*\(")
_RX_SIMD_REDUCE = re.compile(r"\bsimd_(sum|max|min|prefix_inclusive_sum|prefix_exclusive_sum|product)\b")
_RX_FAST_MATH = re.compile(r"\b(fast::|precise::)\w+")
_RX_DEVICE_PARAM = re.compile(r"\b(?:device|constant)\s+(?:const\s+)?\w+[\w<>:]*\s*\*\s*\w+")
_RX_INNER_LOAD = re.compile(r"\b[A-Za-z_]\w*\s*\[[^\]]+\]")  # any `name[expr]` access — very rough


def _max_brace_depth(s: str) -> int:
    """Compute the maximum brace nesting depth in the source. A proxy for
    loop-nest depth when most braces in a kernel body delimit control flow.
    """
    depth = 0
    max_d = 0
    for c in s:
        if c == "{":
            depth += 1
            if depth > max_d:
                max_d = depth
        elif c == "}":
            depth = max(0, depth - 1)
    return max_d


def _analyze_metal_source(source: str | None) -> dict[str, Any] | None:
    if not source:
        return None
    src = _strip_comments(source)

    fn_match = _RX_KERNEL_FN.search(src)

    return {
        "kernel_function_name": fn_match.group(1) if fn_match else None,
        "loop_count": len(_RX_FOR_WHILE.findall(src)),
        "max_brace_depth": _max_brace_depth(src),
        "barrier_count": len(_RX_BARRIER.findall(src)),
        "device_param_count": len(_RX_DEVICE_PARAM.findall(src)),
        "indexed_access_count": len(_RX_INNER_LOAD.findall(src)),
        "has_threadgroup_mem": bool(_RX_THREADGROUP_DECL.search(src)),
        "has_simdgroup_matrix": bool(_RX_SIMDGROUP_MATRIX.search(src)),
        "has_simdgroup_load_store": bool(_RX_SIMDGROUP_LDST.search(src)),
        "has_simd_reduction": bool(_RX_SIMD_REDUCE.search(src)),
        "has_float4": bool(_RX_FLOAT4.search(src)),
        "has_half4": bool(_RX_HALF4.search(src)),
        "has_atomic": bool(_RX_ATOMIC.search(src)),
        "has_unroll_pragma": bool(_RX_UNROLL.search(src)),
        "uses_fast_math": bool(_RX_FAST_MATH.search(src)),
        "source_line_count": source.count("\n") + 1,
    }


# ---------------------------------------------------------------------------
# Roofline sanity gate.
# ---------------------------------------------------------------------------

# Under this SOL fraction, with both axes below this fraction-of-peak, we
# override the roofline classification. The defaults reflect the dead-test
# finding: a 2% SOL "balanced" classification with a "near both ceilings"
# suggest is the single most damaging field in the v1 packet.
_SANITY_SOL_THRESHOLD = 0.05
_SANITY_CEILING_FRACTION = 0.60


def _apply_roofline_sanity(rl: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Return (roofline_dict_with_override_applied, override_reason_or_None)."""
    sol = float(rl.get("sol") or 0.0)
    sol_c = float(rl.get("sol_compute") or 0.0)
    sol_m = float(rl.get("sol_memory") or 0.0)

    if sol >= _SANITY_SOL_THRESHOLD:
        return rl, None
    if sol_c >= _SANITY_CEILING_FRACTION or sol_m >= _SANITY_CEILING_FRACTION:
        return rl, None

    note = (
        f"SOL is {sol*100:.1f}% but neither compute axis ({sol_c*100:.1f}%) nor "
        f"memory axis ({sol_m*100:.1f}%) is saturated. Default roofline label "
        f"would mislead an optimizer here. Common causes when this fires: "
        f"launch / dispatch overhead, instruction stall, low occupancy, "
        f"divergence, sub-resolution timing. Investigate the source_analysis "
        f"block (loop depth, simdgroup usage) before optimizing toward either "
        f"ceiling."
    )

    return (
        {
            **rl,
            "classification": "under_roofline_likely_latency_or_stall",
            "suggest": None,
            "_sanity_override_reason": note,
        },
        note,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shape_bytes(shape, dtype_bytes: int = 4) -> int | None:
    """Return bytes for an input shape, or None if the shape is malformed.

    Defensive against truncated registry parses where `shape` ends up as a
    string fragment (e.g. '(1, 32') instead of a tuple of ints.
    """
    try:
        n = 1
        for d in shape:
            n *= int(d)
        return n * dtype_bytes
    except (TypeError, ValueError):
        return None


def _product(xs) -> int | None:
    try:
        p = 1
        for x in xs:
            p *= int(x)
        return p
    except (TypeError, ValueError):
        return None


def _safe_float(x) -> float:
    """Coerce a registry field to float, tolerating malformed parses."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _chip_id(chip: str | None) -> str:
    from agent_steel.chips import detect_generation
    return detect_generation(chip, fallback="m2")


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
    metal_source: str | None = None,    # NEW in v2 — drives source_analysis
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
    # 2. Dispatch correctness check (unchanged from v1 — this was signal)
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
    # 3. Buffer sanity — emit only when there's real per-buffer info to
    # share (harness captures usually have all-null labels/lengths, which
    # is just noise).
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
    # Suppress the buffer_check section entirely when every entry is informationless.
    if buffer_check and all(
        (b.get("actual_bytes") is None and b.get("label") is None) for b in buffer_check
    ):
        out["buffer_check"] = None
    else:
        out["buffer_check"] = buffer_check or None

    # ------------------------------------------------------------------
    # 4. Timing trust — flags only (the LLM gets the numbers from
    # BenchResult; no point duplicating them here).
    # ------------------------------------------------------------------
    median = getattr(bench, "kernel_ms", None)
    mean = getattr(bench, "kernel_ms_mean", None)
    mtm = (mean / median) if (median and mean) else None
    out["timing_trust"] = {
        "median_ms": median,                     # one anchor value the LLM can latch onto
        "mean_to_median_ratio": (
            round(mtm, 3) if mtm is not None else None
        ),
        "is_sub_resolution": (median is not None and median < 0.005),
        "is_thermally_jittery": (mtm is not None and mtm > 1.5),
    }

    # ------------------------------------------------------------------
    # 5. Roofline + sanity gate + headroom + bottleneck label
    # ------------------------------------------------------------------
    flops = _safe_float(registry_entry.get("flops"))
    bytes_ = _safe_float(registry_entry.get("bytes"))
    chip_g = _chip_id(getattr(bench, "chip", None))
    rl_raw = _roofline.classify(chip_g, flops, bytes_, median or 0.001)
    rl, override_reason = _apply_roofline_sanity(rl_raw)

    sol_c = float(rl_raw.get("sol_compute") or 0.0)
    sol_m = float(rl_raw.get("sol_memory") or 0.0)
    head_c = max(0.0, (1.0 - sol_c) * 100.0)
    head_m = max(0.0, (1.0 - sol_m) * 100.0)
    dominant = "compute" if head_c >= head_m else "memory"

    out["roofline"] = {
        "classification": rl.get("classification"),
        "sol": rl.get("sol"),
        "sol_compute_pct": round(sol_c * 100.0, 2),
        "sol_memory_pct": round(sol_m * 100.0, 2),
        "headroom_compute_pct": round(head_c, 2),
        "headroom_memory_pct": round(head_m, 2),
        "dominant_headroom": dominant,
        "arith_intensity": rl_raw.get("intensity"),
        "ridge_intensity": rl_raw.get("ridge"),
        # `suggest` is None when the sanity gate fires; emit nothing
        # rather than the misleading canned string.
        "suggest": rl.get("suggest"),
        "_sanity_override_reason": override_reason,
    }
    out["bottleneck_label"] = rl.get("classification")

    # ------------------------------------------------------------------
    # 6. Chip ceilings — absolute budgets to ground tile-size recs.
    # ------------------------------------------------------------------
    out["chip_ceilings"] = _chip_ceilings_for(chip_g)

    # ------------------------------------------------------------------
    # 7. Source analysis — what the .metal actually DOES. Drives every
    # code-shape suggestion the LLM might make.
    # ------------------------------------------------------------------
    out["source_analysis"] = _analyze_metal_source(metal_source)

    # ------------------------------------------------------------------
    # 8. Occupancy (trimmed to one-liner)
    # ------------------------------------------------------------------
    tg_mem = (
        (session_record or {}).get("tg_static_mem_bytes")
        if session_record else None
    )
    if tg_mem is None:
        tg_mem = getattr(bench, "tg_mem_bytes", None)
    tg_thr = _product(reg_tg) if reg_tg else None
    out["occupancy_estimates"] = {
        "tg_static_mem_bytes": tg_mem,
        "tg_threads": tg_thr,
        "tg_fill_ratio": (
            round(tg_thr / pso_limit, 3)
            if (tg_thr and pso_limit) else None
        ),
        # one-liner replacement for the v1 essay
        "_note": "Real ALU utilization isn't exposed by Apple's public counter API; infer occupancy from achieved GB/s vs roofline.",
    }

    # ------------------------------------------------------------------
    # 9. History context (unchanged)
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
