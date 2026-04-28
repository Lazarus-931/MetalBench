#!/usr/bin/env python3
"""MetalBench accuracy + speed harness.

    python src/scripts/harness.py <name> [--iters 100] [--warmup 10]
                                          [--seed 0] [--dry-run]
                                          [--capture path.gputrace]

Loads python/metalbench/<name>.py, dispatches build/<name>.metallib via the
host binary, times the MLX reference, compares correctness, prints a JSON
report. Exit code = 0 iff correct.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import mlx.core as mx

# Allow `import mlx_helpers` / `import timing` regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx_helpers as H
from timing import time_mlx, warm_jit


def _allclose(actual, expected, rtol: float, atol: float):
    a = np.asarray(actual).astype(np.float64)
    b = np.asarray(expected).astype(np.float64)
    if a.shape != b.shape:
        return False, float("inf")
    diff = np.abs(a - b)
    tol  = atol + rtol * np.abs(b)
    return bool(np.all(diff <= tol)), float(diff.max()) if a.size else 0.0


# --- dispatch ----------------------------------------------------------------

def _run_host(manifest: dict, manifest_path: Path) -> dict:
    if not H.HOST_BIN.exists():
        raise RuntimeError(f"host binary missing: {H.HOST_BIN}\nrun `make host`")
    manifest_path.write_text(json.dumps(manifest))
    proc = subprocess.run(
        [str(H.HOST_BIN), "--manifest", str(manifest_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"host failed (exit {proc.returncode}):\n{proc.stderr.strip()}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --- main --------------------------------------------------------------------

def evaluate(name: str, *, seed: int, warmup: int, iters: int,
             dry_run: bool = False, capture_path: str | None = None,
             cold_start: bool = False) -> dict:
    b = H.load_baseline(name)
    inputs = b.make_inputs(seed)
    for x in inputs: mx.eval(x)

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
        )
        if dry_run:
            return {"task": name, "dry_run": True, "manifest": manifest}

        k_t = _run_host(manifest, tmp / "manifest.json")

        kernel_outs: list[mx.array] = []
        for o, path in zip(b.outputs, out_paths):
            arr = np.fromfile(path, dtype=H.DTYPE_NP[o.dtype]).reshape(o.shape)
            kernel_outs.append(mx.array(arr))

    warm_jit(b.reference, *inputs)
    ref_outs_raw = b.reference(*inputs)
    ref_outs = list(ref_outs_raw) if isinstance(ref_outs_raw, (list, tuple)) else [ref_outs_raw]
    for r in ref_outs: mx.eval(r)
    mx.synchronize()

    if capture_path:
        with H.capture(capture_path):
            _ = b.reference(*inputs); mx.eval(_); mx.synchronize()

    r_t = time_mlx(b.reference, *inputs, warmup=warmup, iters=iters,
                   cold_start=cold_start)

    if len(kernel_outs) != len(ref_outs):
        raise RuntimeError(
            f"{name}: kernel produced {len(kernel_outs)} outputs, "
            f"reference produced {len(ref_outs)}"
        )

    per_output = []
    for k, r in zip(kernel_outs, ref_outs):
        ok, max_err = _allclose(k, r, b.rtol, b.atol)
        per_output.append({"ok": ok, "max_err": max_err})

    correct = all(o["ok"] for o in per_output)
    speedup = (r_t["median_ms"] / k_t["median_ms"]) if k_t["median_ms"] > 0 else float("inf")

    chip = H.chip_info()
    return {
        "task":             name,
        "chip":             chip,
        "metal_device":     k_t.get("metal_device", k_t.get("device", "unknown")),
        "correct":          correct,
        "speedup":          speedup,
        "kernel_timing":    k_t,
        "reference_timing": r_t,
        "outputs":          per_output,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="metalbench-harness")
    ap.add_argument("name", help="baseline name (matches python/metalbench/<name>.py)")
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--warmup",  type=int, default=10,
                    help="warmup iterations before timing (default 10)")
    ap.add_argument("--iters",   type=int, default=100,
                    help="timed iterations to average over (default 100)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the manifest and print it; don't dispatch")
    ap.add_argument("--capture", default=None,
                    help="record an MLX .gputrace at this path (open in Xcode)")
    ap.add_argument("--cold-start", action="store_true",
                    help="clear MLX kernel cache before timing; measure first-launch latency")
    ap.add_argument("--save",    action="store_true",
                    help="write the result JSON to results/<chip-bucket>/<name>.json")
    args = ap.parse_args(argv)

    # Identify the chip BEFORE doing any work so the user sees what bucket
    # this run will land in. Refusing to proceed on Unknown is too strict
    # (works fine on simulators / future chips) — just call it out.
    chip = H.chip_info()
    print(
        f"[chip] {chip['type']:>8s}  "
        f"name='{chip['name']}'  bucket='{chip['bucket']}'  "
        f"cpu={chip['cpu_cores']}  gpu={chip['gpu_cores']}  "
        f"ram={chip['ram_bytes']/1e9:.0f}GB",
        file=sys.stderr,
    )
    if chip["type"] == "unknown":
        print(
            "[chip] WARNING: chip type unrecognized; results still saved under "
            f"'results/{chip['bucket']}/' but cross-machine comparison may be off.",
            file=sys.stderr,
        )

    result = evaluate(args.name, seed=args.seed, warmup=args.warmup,
                      iters=args.iters, dry_run=args.dry_run,
                      capture_path=args.capture, cold_start=args.cold_start)
    print(json.dumps(result, indent=2))

    if args.save and not args.dry_run:
        bucket = H.bucket_key()
        out_dir = H.RESULTS_ROOT / bucket
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.name}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"saved → {out_path.relative_to(H.REPO_ROOT)}", file=sys.stderr)

    if args.dry_run:
        return 0
    return 0 if result["correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
