"""Shared helpers for MLX baselines and the harness.

Each baseline lives at `mlx/kernels/<set>/<name>.py` and pairs 1:1 with a
kernel at `src/kernels/<set>/<name>.metal` → `build/<name>.metallib`.

Required baseline attributes:
    metal_function : str                 — kernel symbol in the metallib
    threadgroup    : (int, int, int)
    input_bindings : tuple[int, ...]
    outputs        : list[Output]
    make_inputs(seed)         -> list[mx.array]
    reference(*inputs)        -> mx.array | tuple
Optional:
    scalars(inputs)           -> list[Scalar]
    grid(inputs)              -> (int, int, int)
    flops(inputs)  -> int     — for GFLOPS metric
    bytes(inputs)  -> int     — for GB/s metric
    BEST_FOR       : list[str] — chip families this kernel is the best choice for
    rtol, atol     : float
"""
from __future__ import annotations
"""Shared helpers for MLX baselines and the harness.

Use make_element_wise / make_matmul / make_reduction in each baseline file
to avoid boilerplate. They inject grid/scalars/flops/bytes on the module.
"""
import contextlib
import importlib.util
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence, Tuple

import numpy as np
import mlx.core as mx
from pydantic import BaseModel, Field, ValidationError, field_validator

REPO_ROOT     = Path(__file__).resolve().parents[2]
BUILD_DIR     = REPO_ROOT / "build"
HOST_BIN      = BUILD_DIR / "metalbench_host"
BASELINE_ROOT = REPO_ROOT / "mlx" / "kernels"
KERNEL_ROOT   = REPO_ROOT / "src" / "kernels"
RESULTS_ROOT  = REPO_ROOT / "results"
SESSION_PATH  = REPO_ROOT / "session.json"
SETS = ("common", "standard", "full")


def find_kernel_source(name: str) -> Path | None:
    """Locate src/kernels/<set>/<name>.metal."""
    for s in SETS:
        p = KERNEL_ROOT / s / f"{name}.metal"
        if p.exists():
            return p
    return None

DType = str
DTYPE_NP    = {"f32": np.float32, "f16": np.float16, "i32": np.int32, "u32": np.uint32}
DTYPE_BYTES = {"f32": 4, "f16": 2, "i32": 4, "u32": 4}
MX_DTYPE_TAG = {
    mx.float32: "f32", mx.float16: "f16",
    mx.int32:   "i32", mx.uint32:  "u32",
}
_VALID_DTYPES = ("f32", "f16", "i32", "u32")


class Output(BaseModel):
    binding: int = Field(ge=0, lt=32)
    dtype:   str
    shape:   Tuple[int, ...]

    @field_validator("dtype")
    @classmethod
    def _ok(cls, v):
        if v not in _VALID_DTYPES:
            raise ValueError(f"dtype must be {_VALID_DTYPES}, got {v!r}")
        return v


class Scalar(BaseModel):
    binding: int = Field(ge=0, lt=32)
    dtype:   str
    value:   float | int

    @field_validator("dtype")
    @classmethod
    def _ok(cls, v):
        if v not in ("u32", "i32", "f32"):
            raise ValueError(f"scalar dtype must be u32|i32|f32, got {v!r}")
        return v


class BaselineSpec(BaseModel):
    name:           str
    metal_function: str
    threadgroup:    Tuple[int, int, int]
    input_bindings: Tuple[int, ...]
    outputs:        list[Output]
    rtol:           float = 1e-4
    atol:           float = 1e-5
    best_for:       Tuple[str, ...] = ()

    @field_validator("threadgroup")
    @classmethod
    def _tg_pos(cls, v):
        if any(x <= 0 for x in v):
            raise ValueError(f"threadgroup dims must be > 0, got {v}")
        return v

    @field_validator("input_bindings")
    @classmethod
    def _unique(cls, v):
        if len(set(v)) != len(v):
            raise ValueError(f"input_bindings must be unique, got {v}")
        return v


class Baseline:
    def __init__(self, name, module, metallib, spec):
        self.name, self.module, self.metallib, self._spec = name, module, metallib, spec

    @property
    def metal_function(self): return self._spec.metal_function
    @property
    def threadgroup(self):    return self._spec.threadgroup
    @property
    def input_bindings(self): return self._spec.input_bindings
    @property
    def outputs(self):        return self._spec.outputs
    @property
    def rtol(self):           return self._spec.rtol
    @property
    def atol(self):           return self._spec.atol
    @property
    def best_for(self):       return self._spec.best_for

    def make_inputs(self, seed): return list(self.module.make_inputs(seed))
    def reference(self, *inputs): return self.module.reference(*inputs)

    def flops(self, inputs):
        fn = getattr(self.module, "flops", None)
        return int(fn(inputs)) if fn else None

    def bytes(self, inputs):
        fn = getattr(self.module, "bytes", None)
        return int(fn(inputs)) if fn else None

    def scalars(self, inputs):
        fn = getattr(self.module, "scalars", None)
        return list(fn(inputs)) if fn else []

    def grid(self, inputs):
        fn = getattr(self.module, "grid", None)
        if fn:
            return tuple(fn(inputs))
        shp = list(self.outputs[0].shape) + [1, 1, 1]
        return (int(shp[0]), int(shp[1]), int(shp[2]))


def _find_baseline(name):
    for s in SETS:
        p = BASELINE_ROOT / s / f"{name}.py"
        if p.exists():
            return p, s
    raise FileNotFoundError(
        f"baseline not found: searched mlx/kernels/{{{','.join(SETS)}}}/{name}.py"
    )


def load_baseline(name):
    path, _set = _find_baseline(name)

    metallib = BUILD_DIR / f"{name}.metallib"
    if not metallib.exists():
        raise FileNotFoundError(
            f"metallib not built: {metallib}\n"
            f"expected source at {KERNEL_ROOT / _set / f'{name}.metal'}; run `make`."
        )

    # Load Model class from .py file
    mod_spec = importlib.util.spec_from_file_location(f"_mb_{name}", path)
    if mod_spec is None or mod_spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)

    ModelCls = getattr(mod, "Model", None)
    if ModelCls is None:
        raise AttributeError(f"{name}.py: missing Model class")

    # Load metadata from registry
    reg_spec = importlib.util.spec_from_file_location(
        "_mb_registry", BASELINE_ROOT / _set / "registry.py")
    if reg_spec is None or reg_spec.loader is None:
        raise ImportError("cannot load registry.py")
    reg_mod = importlib.util.module_from_spec(reg_spec)
    reg_spec.loader.exec_module(reg_mod)
    meta = reg_mod.REGISTRY.get(name)
    if meta is None:
        raise KeyError(f"no registry entry for '{name}'")

    # Auto-generate get_inputs from registry-provided input shapes
    input_shapes = meta.get("input_shapes")
    if input_shapes is None:
        # Fallback: single input of output_shape
        input_shapes = [meta["output_shape"]]

    def _get_inputs():
        return [mx.random.normal(s, dtype=mx.float32) for s in input_shapes]

    mod.get_inputs = _get_inputs
    mod.get_init_inputs = lambda: []

    def _make_inputs(seed):
        mx.random.seed(seed)
        return list(mod.get_inputs())

    mod.make_inputs = _make_inputs

    model_inst = ModelCls()
    def _reference(*inputs):
        return model_inst.forward(*inputs)

    mod.reference = _reference

    # Convert registry dicts to Output/Scalar objects
    n_in = len(meta["input_shapes"])
    out_shape = meta["output_shape"]
    raw_outputs = [dict(binding=2 if n_in == 2 else 1, dtype="f32", shape=out_shape)]
    outputs = [Output(**o) if isinstance(o, dict) else o for o in raw_outputs]
    raw_scalars = meta["scalars"]
    scalars = [Scalar(**s) if isinstance(s, dict) else s for s in raw_scalars]

    try:
        spec = BaselineSpec(
            name=name,
            metal_function=meta["metal_function"],
            threadgroup=tuple(meta["threadgroup"]),
            input_bindings=tuple(meta["input_bindings"]),
            outputs=outputs,
            rtol=meta.get("rtol", 1e-4),
            atol=meta.get("atol", 1e-5),
            best_for=tuple(meta.get("BEST_FOR", ())),
        )
    except ValidationError as e:
        raise ValueError(f"registry entry for '{name}' invalid:\n{e}") from e

    # Override scalars/grid/flops/bytes from registry
    mod.scalars = lambda inputs: scalars
    mod.grid = lambda inputs: meta["grid"]
    mod.flops = lambda inputs: meta["flops"]
    mod.bytes = lambda inputs: meta["bytes"]

    return Baseline(name=name, module=mod, metallib=metallib, spec=spec)


def build_manifest(b, inputs, *, input_paths, output_paths, warmup, iters):
    if len(input_paths) != len(b.input_bindings):
        raise ValueError("input_paths must match input_bindings")
    if len(output_paths) != len(b.outputs):
        raise ValueError("output_paths must match outputs")

    buffers = []
    for binding, path in zip(b.input_bindings, input_paths):
        buffers.append({"binding": int(binding), "role": "input", "path": str(path)})
    for o, path in zip(b.outputs, output_paths):
        buffers.append({
            "binding": int(o.binding), "role": "output",
            "path": str(path),
            "size": int(np.prod(o.shape)) * DTYPE_BYTES[o.dtype],
        })
    for s in b.scalars(inputs):
        buffers.append({
            "binding": int(s.binding), "role": "scalar",
            "dtype": s.dtype, "value": s.value,
        })

    return {
        "function":    b.metal_function,
        "metallib":    str(b.metallib),
        "buffers":     buffers,
        "grid":        list(b.grid(inputs)),
        "threadgroup": list(b.threadgroup),
        "warmup":      int(warmup),
        "iters":       int(iters),
    }


# --- chip detection (mirrors src/metal_scripts/setup.cpp) -------------------

_CHIP_TYPES = [
    ("M4 Max", "m4_max"), ("M4 Pro", "m4_pro"), ("M4", "m4"),
    ("M3 Ultra", "m3_ultra"), ("M3 Max", "m3_max"), ("M3 Pro", "m3_pro"), ("M3", "m3"),
    ("M2 Ultra", "m2_ultra"), ("M2 Max", "m2_max"), ("M2 Pro", "m2_pro"), ("M2", "m2"),
    ("M1 Ultra", "m1_ultra"), ("M1 Max", "m1_max"), ("M1 Pro", "m1_pro"), ("M1", "m1"),
]


def _sysctl(key):
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return ""


def chip_info():
    name = _sysctl("machdep.cpu.brand_string") or "unknown"
    chip_type = next((tag for needle, tag in _CHIP_TYPES if needle in name), "unknown")

    try: cpu_cores = int(_sysctl("hw.physicalcpu") or 0)
    except Exception: cpu_cores = 0
    try: ram_bytes = int(_sysctl("hw.memsize") or 0)
    except Exception: ram_bytes = 0

    gpu_cores = 0
    try:
        sp = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], text=True, timeout=5)
        import re
        m = re.search(r"Total Number of Cores:\s*(\d+)", sp)
        if m: gpu_cores = int(m.group(1))
    except Exception:
        pass

    return {
        "type":      chip_type,
        "name":      name,
        "bucket":    bucket_key_from_name(name),
        "cpu_cores": cpu_cores,
        "gpu_cores": gpu_cores,
        "ram_bytes": ram_bytes,
    }


def bucket_key_from_name(name):
    s = "".join((c.lower() if c.isalnum() else "-") for c in name)
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-") or "unknown"


def bucket_key():
    return chip_info()["bucket"]


# --- MLX device + capture helpers -------------------------------------------

def device_info():
    info: dict = {"machine": platform.machine(), "os": f"{platform.system()} {platform.release()}"}
    for key, label in [
        ("machdep.cpu.brand_string", "cpu_brand"),
        ("hw.model", "hw_model"),
        ("hw.memsize", "memory_bytes"),
    ]:
        v = _sysctl(key)
        if v: info[label] = v
    try:
        info["mlx_metal_available"] = bool(mx.metal.is_available())
        info["mlx_device_info"]     = mx.metal.device_info()
    except Exception as e:
        info["mlx_error"] = repr(e)
    return info


@contextlib.contextmanager
def capture(path):
    """Record a Metal GPU trace of MLX dispatches. Open the .gputrace in Xcode."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    mx.metal.start_capture(str(p))
    try: yield p
    finally: mx.metal.stop_capture()


def array_summary(arr):
    if isinstance(arr, mx.array):
        mx.eval(arr)
        np_arr = np.asarray(arr); dtype = str(arr.dtype)
    else:
        np_arr = np.asarray(arr); dtype = str(np_arr.dtype)
    flat = np_arr.astype(np.float64, copy=False).ravel()
    return {
        "shape": tuple(np_arr.shape),
        "dtype": dtype,
        "size":  int(np_arr.size),
        "nan":   int(np.isnan(flat).sum()),
        "inf":   int(np.isinf(flat).sum()),
        "min":   float(np.nanmin(flat)) if np_arr.size else None,
        "max":   float(np.nanmax(flat)) if np_arr.size else None,
        "mean":  float(np.nanmean(flat)) if np_arr.size else None,
    }


def mx_to_np_bytes(arr):
    tag = MX_DTYPE_TAG.get(arr.dtype)
    if tag is None:
        raise ValueError(f"unsupported dtype {arr.dtype}; extend MX_DTYPE_TAG")
    mx.eval(arr)
    return np.asarray(arr, dtype=DTYPE_NP[tag]), tag


# ---------------------------------------------------------------------------
# Shared boilerplate factories — call at module level in each baseline.
# ---------------------------------------------------------------------------

def element_wise_spec(metal_func, N_el, grid_size=64 * 1024, rtol=1e-3, atol=1e-3,
                      output_shape=(16, 16384), flops_mul=5):
    """Return a dict of metadata for element-wise kernels."""
    return dict(
        metal_function=metal_func,
        threadgroup=(1024, 1, 1),
        input_bindings=(0,),
        outputs=[Output(binding=1, dtype="f32", shape=output_shape)],
        rtol=rtol, atol=atol,
        scalars=lambda inputs: [
            Scalar(binding=2, dtype="u32", value=N_el),
            Scalar(binding=3, dtype="u32", value=grid_size),
        ],
        grid=lambda inputs: (grid_size, 1, 1),
        flops=lambda inputs: N_el * flops_mul,
        bytes=lambda inputs: 2 * N_el * 4,
    )


def matmul_spec(metal_func, M, N, K, BM=64, BN=64, BK=16, TG_THREADS=256,
                rtol=1e-3, atol=1e-3):
    """Return a dict of metadata for matmul kernels."""
    return dict(
        metal_function=metal_func,
        threadgroup=(TG_THREADS, 1, 1),
        input_bindings=(0, 1),
        outputs=[Output(binding=2, dtype="f32", shape=(M, N))],
        rtol=rtol, atol=atol,
        scalars=lambda inputs: [
            Scalar(binding=3, dtype="u32", value=M),
            Scalar(binding=4, dtype="u32", value=N),
            Scalar(binding=5, dtype="u32", value=K),
        ],
        grid=lambda inputs: ((N // BN) * TG_THREADS, M // BM, 1),
        flops=lambda inputs: 2 * M * N * K,
        bytes=lambda inputs: 4 * (M * K + K * N + M * N),
    )


def batched_matmul_spec(metal_func, B, M, N, K, BM=64, BN=64, BK=16,
                        TG_THREADS=256, rtol=1e-3, atol=1e-3):
    """Return a dict of metadata for batched matmul kernels."""
    return dict(
        metal_function=metal_func,
        threadgroup=(TG_THREADS, 1, 1),
        input_bindings=(0, 1),
        outputs=[Output(binding=2, dtype="f32", shape=(B, M, N))],
        rtol=rtol, atol=atol,
        scalars=lambda inputs: [
            Scalar(binding=3, dtype="u32", value=M),
            Scalar(binding=4, dtype="u32", value=N),
            Scalar(binding=5, dtype="u32", value=K),
        ],
        grid=lambda inputs: ((N // BN) * TG_THREADS, (M // BM) * B, 1),
        flops=lambda inputs: B * 2 * M * N * K,
        bytes=lambda inputs: B * 4 * (M * K + K * N + M * N),
    )
