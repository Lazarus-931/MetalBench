from __future__ import annotations
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import mlx.core as mx

from .task import Task, OutputSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_BIN = REPO_ROOT / "build" / "metalbench_host"

DTYPE_NP = {"f32": np.float32, "i32": np.int32, "u32": np.uint32}
DTYPE_BYTES = {"f32": 4, "i32": 4, "u32": 4}

# mx dtype -> manifest dtype string
MX_DTYPE_TAG = {
    mx.float32: "f32",
    mx.int32:   "i32",
    mx.uint32:  "u32",
}


def _mx_to_np(arr: mx.array) -> tuple[np.ndarray, str]:
    tag = MX_DTYPE_TAG.get(arr.dtype)
    if tag is None:
        raise ValueError(
            f"unsupported input dtype {arr.dtype}; extend MX_DTYPE_TAG in host.py"
        )
    mx.eval(arr)
    return np.asarray(arr, dtype=DTYPE_NP[tag]), tag


def run_kernel(
    task: Task,
    inputs: Sequence[mx.array],
    *,
    warmup: int = 5,
    iters: int = 50,
) -> tuple[list[mx.array], dict]:
    """Dispatch one task on the GPU via the C++ host. Returns (outputs, timings)."""
    if not HOST_BIN.exists():
        raise RuntimeError(
            f"host binary not built: {HOST_BIN}\n"
            f"build it with `make host` from {REPO_ROOT}"
        )

    metallib = Path(task.metallib)
    if not metallib.is_absolute():
        metallib = (REPO_ROOT / metallib).resolve()
    if not metallib.exists():
        raise RuntimeError(f"metallib not found: {metallib} (run `make kernels`)")

    if len(inputs) != len(task.input_bindings):
        raise ValueError(
            f"task {task.name}: {len(inputs)} inputs but "
            f"{len(task.input_bindings)} input_bindings"
        )

    with tempfile.TemporaryDirectory(prefix="mb_") as tmp:
        tmp = Path(tmp)
        buffers: list[dict] = []

        for binding, arr in zip(task.input_bindings, inputs):
            np_arr, _tag = _mx_to_np(arr)
            path = tmp / f"in_{binding}.bin"
            np_arr.tofile(path)
            buffers.append({"binding": int(binding), "role": "input", "path": str(path)})

        out_paths: dict[int, tuple[Path, OutputSpec]] = {}
        for o in task.outputs:
            n = int(np.prod(o.shape))
            size = n * DTYPE_BYTES[o.dtype]
            path = tmp / f"out_{o.binding}.bin"
            out_paths[o.binding] = (path, o)
            buffers.append({
                "binding": int(o.binding),
                "role": "output",
                "path": str(path),
                "size": size,
            })

        for s in task.scalars_fn(inputs):
            buffers.append({
                "binding": int(s.binding),
                "role": "scalar",
                "dtype": s.dtype,
                "value": s.value,
            })

        manifest = {
            "function":    task.function,
            "metallib":    str(metallib),
            "buffers":     buffers,
            "grid":        list(task.grid_fn(inputs)),
            "threadgroup": list(task.threadgroup),
            "warmup":      int(warmup),
            "iters":       int(iters),
        }
        manifest_path = tmp / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        proc = subprocess.run(
            [str(HOST_BIN), "--manifest", str(manifest_path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"host failed (exit {proc.returncode}):\n{proc.stderr.strip()}"
            )

        last_line = proc.stdout.strip().splitlines()[-1]
        timings = json.loads(last_line)

        out_arrays: list[mx.array] = []
        for o in task.outputs:
            path, spec = out_paths[o.binding]
            np_out = np.fromfile(path, dtype=DTYPE_NP[spec.dtype]).reshape(spec.shape)
            out_arrays.append(mx.array(np_out))

        return out_arrays, timings
