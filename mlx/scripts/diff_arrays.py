#!/usr/bin/env python3
"""Diff a kernel output against an MLX reference. Surfaces the worst mismatches.

Usage:
    diff_arrays.py kernel_out.bin reference.bin --shape 1024 1024 --dtype f32
    diff_arrays.py a.npy b.npy --top 20

When correctness fails, this is what tells you *where* the kernel diverges —
e.g. mismatches concentrated on tile boundaries, last column, NaN propagation.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

DTYPE = {"f32": np.float32, "i32": np.int32, "u32": np.uint32, "f16": np.float16}


def _load(path: str, shape, dtype) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    if shape is None or dtype is None:
        raise SystemExit(f"{path}: raw binary requires --shape and --dtype")
    arr = np.fromfile(p, dtype=DTYPE[dtype])
    return arr.reshape(shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("actual",   help="kernel output (.npy or raw binary)")
    ap.add_argument("expected", help="MLX reference (.npy or raw binary)")
    ap.add_argument("--shape", type=int, nargs="+", default=None,
                    help="required for raw binary inputs")
    ap.add_argument("--dtype", choices=DTYPE.keys(), default=None,
                    help="required for raw binary inputs")
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--top",  type=int, default=10,
                    help="show this many worst mismatches")
    args = ap.parse_args()

    a = _load(args.actual,   args.shape, args.dtype).astype(np.float64)
    b = _load(args.expected, args.shape, args.dtype).astype(np.float64)

    if a.shape != b.shape:
        print(f"SHAPE MISMATCH: actual {a.shape} vs expected {b.shape}")
        return 2

    diff = np.abs(a - b)
    rel  = diff / (np.abs(b) + 1e-12)
    tol  = args.atol + args.rtol * np.abs(b)
    bad  = diff > tol

    print(f"shape:        {a.shape}  ({a.size} elements)")
    print(f"actual nan:   {int(np.isnan(a).sum())}    inf: {int(np.isinf(a).sum())}")
    print(f"expected nan: {int(np.isnan(b).sum())}    inf: {int(np.isinf(b).sum())}")
    print(f"max abs err:  {float(diff.max()):.6e}")
    print(f"mean abs err: {float(diff.mean()):.6e}")
    print(f"max rel err:  {float(rel.max()):.6e}")
    print(f"mismatches:   {int(bad.sum())} / {a.size}  "
          f"({100.0 * bad.sum() / a.size:.4f}%)")
    print(f"tolerance:    atol={args.atol} rtol={args.rtol}")

    if bad.any() and args.top > 0:
        flat_idx = np.argsort(-diff.ravel())[: args.top]
        print(f"\ntop {min(args.top, int(bad.sum()))} mismatches:")
        print(f"  {'index':<24} {'actual':>14} {'expected':>14} {'abs_err':>12}")
        for fi in flat_idx:
            if diff.ravel()[fi] == 0:
                break
            idx = np.unravel_index(fi, a.shape)
            print(f"  {str(idx):<24} {a[idx]:>14.6g} {b[idx]:>14.6g} {diff[idx]:>12.3e}")

    return 0 if not bad.any() else 1


if __name__ == "__main__":
    sys.exit(main())
