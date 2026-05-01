#!/usr/bin/env python3
"""MetalBench harness: dispatch a kernel, time it, grade it.

    python src/mlx_scripts/harness.py <name> [--target speed|compute|memory|stable|balanced]
                                              [--iters 100] [--warmup 10] [--seed 0]
                                              [--save] [--cold-start] [--dry-run]
                                              [--capture out.gputrace]
"""
from __future__ import annotations
import argparse
import datetime as _dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx_helpers as H
from timing import time_mlx, warm_jit


def _allclose(actual, expected, rtol, atol):
    a = np.asarray(actual).astype(np.float64)
    b = np.asarray(expected).astype(np.float64)
    if a.shape != b.shape:
        return False, float("inf")
    diff = np.abs(a - b)
    tol  = atol + rtol * np.abs(b)
    return bool(np.all(diff <= tol)), float(diff.max()) if a.size else 0.0


def _run_host(manifest, manifest_path):
    if not H.HOST_BIN.exists():
        raise RuntimeError(f"host binary missing: {H.HOST_BIN}\nrun `make host`")
    manifest_path.write_text(json.dumps(manifest))
    proc = subprocess.run(
        [str(H.HOST_BIN), "--manifest", str(manifest_path)],
        capture_output=True, text=True,
    )
    # Forward host's stderr banner ([host] device, [host] kernel ready) up to user.
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"host failed (exit {proc.returncode})")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def evaluate(name, *, seed, warmup, iters, dry_run=False,
             capture_path=None, cold_start=False,
             profile=False, capture_host=None):
    b = H.load_baseline(name)
    inputs = b.make_inputs(seed)
    for x in inputs: mx.eval(x)

    chip = H.chip_info()

    with tempfile.TemporaryDirectory(prefix="mb_") as tmp:
        tmp = Path(tmp)
        in_paths  = [tmp / f"in_{bi}.bin"  for bi in b.input_bindings]
        out_paths = [tmp / f"out_{o.binding}.bin" for o in b.outputs]

        for path, arr in zip(in_paths, inputs):
            np_arr, _ = H.mx_to_np_bytes(arr)
            np_arr.tofile(path)

        manifest = H.build_manifest(
            b, inputs, input_paths=in_paths, output_paths=out_paths,
            warmup=warmup, iters=iters,
            chip_type=chip["type"],
            profile=profile,
            capture_host=capture_host,
        )
        if dry_run:
            return {"task": name, "dry_run": True, "manifest": manifest}

        k_t = _run_host(manifest, tmp / "manifest.json")

        kernel_outs = [
            mx.array(np.fromfile(p, dtype=H.DTYPE_NP[o.dtype]).reshape(o.shape))
            for o, p in zip(b.outputs, out_paths)
        ]

    warm_jit(b.reference, *inputs)
    ref_out = b.reference(*inputs)
    ref_outs = list(ref_out) if isinstance(ref_out, (list, tuple)) else [ref_out]
    for r in ref_outs: mx.eval(r)
    mx.synchronize()

    if capture_path:
        with H.capture(capture_path):
            _ = b.reference(*inputs); mx.eval(_); mx.synchronize()

    r_t = time_mlx(b.reference, *inputs, warmup=warmup, iters=iters,
                   cold_start=cold_start)

    if len(kernel_outs) != len(ref_outs):
        raise RuntimeError(
            f"{name}: kernel={len(kernel_outs)} outputs, ref={len(ref_outs)}"
        )

    per_output = []
    for k, r in zip(kernel_outs, ref_outs):
        ok, max_err = _allclose(k, r, b.rtol, b.atol)
        per_output.append({"ok": ok, "max_err": max_err})

    correct = all(o["ok"] for o in per_output)
    speedup = (r_t["median_ms"] / k_t["median_ms"]) if k_t["median_ms"] > 0 else float("inf")

    metrics = {"speedup": speedup}
    median_s = k_t["median_ms"] / 1000.0
    if median_s > 0:
        if (f := b.flops(inputs)) is not None:
            metrics["gflops"] = f / median_s / 1e9
        if (nb := b.bytes(inputs)) is not None:
            metrics["gbps"] = nb / median_s / 1e9
        if f is not None and nb is not None and nb > 0:
            metrics["arith_intensity"] = f / nb

    # Stability proxy from min/median/mean (we don't keep the full timing list).
    mean_ms = k_t["mean_ms"]
    cv = abs(k_t["median_ms"] - mean_ms) / mean_ms if mean_ms > 0 else 0.0
    metrics["stability"] = max(0.0, 1.0 - cv)

    result = {
        "task":             name,
        "chip":             chip,
        "metal_device":     k_t.get("metal_device", "unknown"),
        "tg_static_mem_bytes":     k_t.get("tg_static_mem_bytes"),
        "pso_max_threads_per_tg":  k_t.get("pso_max_threads_per_tg"),
        "correct":          correct,
        "metrics":          metrics,
        "speedup":          speedup,
        "kernel_timing":    k_t,
        "reference_timing": r_t,
        "outputs":          per_output,
    }
    if "available_counter_sets" in k_t:
        result["available_counter_sets"] = k_t["available_counter_sets"]
    if "profiling" in k_t:
        result["profiling"] = k_t["profiling"]
    return result


# Each target picks ONE primary metric to optimize. Higher = better.
TARGETS = {
    "speed":    lambda m: m.get("speedup",   0.0),
    "compute":  lambda m: m.get("gflops",    0.0),
    "memory":   lambda m: m.get("gbps",      0.0),
    "stable":   lambda m: m.get("stability", 0.0),
    "balanced": lambda m: (
        0.5 * m.get("speedup", 0.0)
      + 0.3 * (m.get("gflops", 0.0) / 1000.0)
      + 0.2 * m.get("stability", 0.0)
    ),
}


def grade(metrics, target):
    fn = TARGETS.get(target)
    if fn is None:
        raise SystemExit(f"unknown target '{target}'; pick {list(TARGETS)}")
    return float(fn(metrics))


def update_session(name, result, target, iters):
    """Maintain session.json — best run per (chip, kernel).

    Best = lowest kernel median_ms (chip-side time, independent of MLX
    warmup variance). Stores the entire .metal source so the recorded
    best run is reproducible from the file alone.

    Skips the update if the run was incorrect.
    """
    if not result.get("correct"):
        return False, "skipped (incorrect)"

    bucket   = result["chip"]["bucket"]
    new_ms   = result["kernel_timing"]["median_ms"]

    src_path = H.find_kernel_source(name)
    src_text = src_path.read_text() if src_path else None

    session = {}
    if H.SESSION_PATH.exists():
        try:
            session = json.loads(H.SESSION_PATH.read_text())
        except Exception:
            session = {}

    chip_section = session.setdefault(bucket, {})
    prev = chip_section.get(name)
    prev_ms = prev["best_time_ms"] if prev else None

    if prev_ms is not None and new_ms >= prev_ms:
        return False, f"current best {prev_ms:.3f} ms holds (this run {new_ms:.3f} ms)"

    chip_section[name] = {
        "best_time_ms":      new_ms,
        "speedup_vs_mlx":    result["metrics"]["speedup"],
        "gflops":            result["metrics"].get("gflops"),
        "gbps":              result["metrics"].get("gbps"),
        "arith_intensity":   result["metrics"].get("arith_intensity"),
        "stability":         result["metrics"]["stability"],
        "tg_static_mem_bytes":    result.get("tg_static_mem_bytes"),
        "pso_max_threads_per_tg": result.get("pso_max_threads_per_tg"),
        "iters":             iters,
        "target":            target,
        "max_err":           max((o["max_err"] for o in result["outputs"]), default=0.0),
        "metal_source_path": str(src_path.relative_to(H.REPO_ROOT)) if src_path else None,
        "metal_source":      src_text,
        "updated_at":        _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }

    H.SESSION_PATH.write_text(json.dumps(session, indent=2) + "\n")
    delta = "" if prev_ms is None else f" (was {prev_ms:.3f} ms, Δ {prev_ms - new_ms:+.3f})"
    return True, f"new best {new_ms:.3f} ms{delta}"


def _bytes_human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024: return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.0f}TB"


def _print_report(result, target, score):
    """Human-readable report. Replaces JSON dump on stdout."""
    chip   = result["chip"]
    m      = result["metrics"]
    k_t    = result["kernel_timing"]
    r_t    = result["reference_timing"]
    outs   = result["outputs"]

    line = "─" * 72
    print(line)
    print(f"  {result['task']}    target={target}   score={score:.3f}")
    print(line)
    print(f"  device      : {chip['name']} ({chip['type']})  "
          f"{chip['cpu_cores']} CPU / {chip['gpu_cores']} GPU / "
          f"{chip['ram_bytes']/1e9:.0f} GB")
    if result.get("tg_static_mem_bytes") is not None:
        print(f"  occupancy   : tg_mem={_bytes_human(result['tg_static_mem_bytes'])}  "
              f"max_thr/tg={result['pso_max_threads_per_tg']}")
    if result.get("available_counter_sets"):
        csets = ", ".join(result["available_counter_sets"])
        print(f"  counters    : {csets}")
    print()
    status = "✓ correct " if result["correct"] else "✗ INCORRECT"
    max_err = max((o["max_err"] for o in outs), default=0.0)
    print(f"  correctness : {status}    max_err={max_err:.3e}")
    print(f"  speedup     : {m.get('speedup', 0):.2f}× vs MLX")
    print(f"  kernel      : {k_t['median_ms']:.3f} ms  "
          f"(min {k_t['min_ms']:.3f}, mean {k_t['mean_ms']:.3f}, n={k_t['iters']})")
    print(f"  mlx ref     : {r_t['median_ms']:.3f} ms  "
          f"(min {r_t['min_ms']:.3f}, mean {r_t['mean_ms']:.3f}, n={r_t['iters']})")
    if "gflops" in m:
        print(f"  compute     : {m['gflops']:.1f} GFLOPS")
    if "gbps" in m:
        print(f"  bandwidth   : {m['gbps']:.1f} GB/s")
    if "arith_intensity" in m:
        print(f"  arith int.  : {m['arith_intensity']:.1f} FLOPs/byte")
    print(f"  stability   : {m['stability']:.2f}  (1.0 = perfectly consistent)")
    prof = result.get("profiling", {})
    if prof.get("counters_available"):
        print(f"  profiling   : {prof['counter_set']}")
        for s in prof.get("samples", []):
            print(f"    {s['name']:40s} {s['value']}")
    print()
    print(f"  {'target':>10s}   {'score':>10s}")
    print(f"  {'-'*10}   {'-'*10}")
    for t in ("speed", "compute", "memory", "stable", "balanced"):
        s = grade(m, t)
        print(f"  {t:>10s}   {s:10.3f}")
    print(line)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="metalbench-harness")
    ap.add_argument("name")
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--warmup",     type=int, default=10)
    ap.add_argument("--iters",      type=int, default=200)
    ap.add_argument("--dry-run",    action="store_true")
    ap.add_argument("--capture",    default=None,
                    help="record an MLX .gputrace at this path")
    ap.add_argument("--capture-host", default=None,
                    help="record a .gputrace of the host Metal dispatch at this path")
    ap.add_argument("--profile",    action="store_true",
                    help="enable GPU counter sampling via MTLCounterSet (adds one extra dispatch)")
    ap.add_argument("--cold-start", action="store_true",
                    help="clear MLX kernel cache before timing")
    ap.add_argument("--save",       action="store_true",
                    help="save full result JSON to results/<chip>/<name>.json")
    ap.add_argument("--no-session", action="store_true",
                    help="don't update session.json")
    ap.add_argument("--target",     default="speed", choices=list(TARGETS.keys()))
    args = ap.parse_args(argv)

    chip = H.chip_info()
    print(f"[chip] {chip['type']:>8s}  '{chip['name']}'  "
          f"cpu={chip['cpu_cores']}  gpu={chip['gpu_cores']}  "
          f"ram={chip['ram_bytes']/1e9:.0f}GB", file=sys.stderr)

    if chip["type"] == "unknown":
        print(f"[chip] WARN: unrecognized chip; bucket='{chip['bucket']}'",
              file=sys.stderr)

    try:
        bf = H.load_baseline(args.name).best_for
        if bf and "all" not in bf and chip["type"] not in bf:
            print(f"[bench] kernel `{args.name}` BEST_FOR={list(bf)}; "
                  f"current chip '{chip['type']}' not in that list",
                  file=sys.stderr)
        elif bf:
            print(f"[bench] kernel `{args.name}` BEST_FOR={list(bf)}",
                  file=sys.stderr)
    except Exception:
        pass

    result = evaluate(args.name, seed=args.seed, warmup=args.warmup,
                      iters=args.iters, dry_run=args.dry_run,
                      capture_path=args.capture, cold_start=args.cold_start,
                      profile=args.profile, capture_host=args.capture_host)

    if args.dry_run:
        # For dry run, print the manifest as text dump (it's the only artifact).
        print(json.dumps(result["manifest"], indent=2))
        return 0

    score = grade(result["metrics"], args.target)
    result["target"] = args.target
    result["score"]  = score
    _print_report(result, args.target, score)

    if args.save:
        bucket = H.bucket_key()
        out_dir = H.RESULTS_ROOT / bucket
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.name}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"saved → {out_path.relative_to(H.REPO_ROOT)}", file=sys.stderr)

    if not args.no_session:
        updated, why = update_session(args.name, result, args.target, args.iters)
        rel = H.SESSION_PATH.relative_to(H.REPO_ROOT)
        prefix = f"updated {rel}" if updated else f"{rel} unchanged"
        print(f"{prefix} [{result['chip']['bucket']}/{args.name}]: {why}",
              file=sys.stderr)

    return 0 if result["correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
