"""Shared helpers for MLX baseline scripts and the harness.

Each baseline lives at `python/metalbench/<name>.py` and pairs 1:1 with a
kernel at `src/kernels/<name>/kernel.metal` → `build/<name>.metallib`.

A baseline module is expected to expose:
    metal_function : str                   # kernel symbol in the metallib
    threadgroup    : (int, int, int)
    input_bindings : tuple[int, ...]       # binding index per input, in order
    outputs        : list[Output]          # what the host allocates + reads back
    make_inputs(seed) -> list[mx.array]
    reference(*inputs) -> mx.array | tuple[mx.array, ...]
optional:
    scalars(inputs) -> list[Scalar]        # default: []
    grid(inputs)    -> (int, int, int)     # default: outputs[0].shape padded to 3D
    rtol, atol      : float                # default: 1e-4, 1e-5

"""
from __future__ import annotations
import contextlib
import importlib.util
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence, Tuple

import numpy as np
import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR    = REPO_ROOT / "build"
HOST_BIN     = BUILD_DIR / "metalbench_host"
BASELINE_ROOT = REPO_ROOT / "python" / "kernels"   # contains common/, standard/, full/
KERNEL_ROOT   = REPO_ROOT / "src" / "kernels"      # ditto
RESULTS_ROOT  = REPO_ROOT / "results"
SETS = ("common", "standard", "full")

DType = str  # "f32" | "f16" | "i32" | "u32"

DTYPE_NP = {"f32": np.float32, "f16": np.float16, "i32": np.int32, "u32": np.uint32}
DTYPE_BYTES = {"f32": 4, "f16": 2, "i32": 4, "u32": 4}

MX_DTYPE_TAG = {
    mx.float32: "f32",
    mx.float16: "f16",
    mx.int32:   "i32",
    mx.uint32:  "u32",
}


@dataclass
class Output:
    binding: int
    dtype: DType
    shape: Sequence[int]


@dataclass
class Scalar:
    binding: int
    dtype: DType    # "u32" | "i32" | "f32"
    value: float | int


@dataclass
class Baseline:
    """Resolved baseline — module + derived paths."""
    name: str
    module: Any
    metallib: Path

    @property
    def metal_function(self) -> str:    return self.module.metal_function
    @property
    def threadgroup(self) -> Tuple[int, int, int]: return tuple(self.module.threadgroup)
    @property
    def input_bindings(self) -> Tuple[int, ...]:   return tuple(self.module.input_bindings)
    @property
    def outputs(self) -> list[Output]:  return list(self.module.outputs)
    @property
    def rtol(self) -> float:            return float(getattr(self.module, "rtol", 1e-4))
    @property
    def atol(self) -> float:            return float(getattr(self.module, "atol", 1e-5))

    def make_inputs(self, seed: int):   return list(self.module.make_inputs(seed))
    def reference(self, *inputs):       return self.module.reference(*inputs)

    def scalars(self, inputs) -> list[Scalar]:
        fn = getattr(self.module, "scalars", None)
        return list(fn(inputs)) if fn else []

    def grid(self, inputs) -> Tuple[int, int, int]:
        fn = getattr(self.module, "grid", None)
        if fn:
            return tuple(fn(inputs))
        # default: the first output's shape padded to 3D
        shp = list(self.outputs[0].shape) + [1, 1, 1]
        return (int(shp[0]), int(shp[1]), int(shp[2]))


def _find_baseline_path(name: str) -> tuple[Path, str]:
    """Search python/kernels/<set>/<name>.py across common/standard/full."""
    for s in SETS:
        p = BASELINE_ROOT / s / f"{name}.py"
        if p.exists():
            return p, s
    raise FileNotFoundError(
        f"baseline not found: searched python/kernels/{{{','.join(SETS)}}}/{name}.py"
    )


def load_baseline(name: str) -> Baseline:
    """Locate python/kernels/<set>/<name>.py and the matching metallib."""
    path, _set = _find_baseline_path(name)

    metallib = BUILD_DIR / f"{name}.metallib"
    if not metallib.exists():
        raise FileNotFoundError(
            f"metallib not built: {metallib}\n"
            f"expected source at {KERNEL_ROOT / _set / f'{name}.metal'}; run `make`."
        )

    spec = importlib.util.spec_from_file_location(f"_mb_baseline_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for attr in ("metal_function", "threadgroup", "input_bindings",
                 "outputs", "make_inputs", "reference"):
        if not hasattr(mod, attr):
            raise AttributeError(f"{name}.py missing required attribute `{attr}`")

    return Baseline(name=name, module=mod, metallib=metallib)


def build_manifest(b: Baseline, inputs: Sequence[mx.array],
                   *, input_paths: Sequence[Path], output_paths: Sequence[Path],
                   warmup: int, iters: int) -> dict:
    """Compose the JSON manifest the host consumes. Caller writes the files."""
    if len(input_paths) != len(b.input_bindings):
        raise ValueError("input_paths must match input_bindings")
    if len(output_paths) != len(b.outputs):
        raise ValueError("output_paths must match outputs")

    buffers: list[dict] = []
    for binding, path in zip(b.input_bindings, input_paths):
        buffers.append({"binding": int(binding), "role": "input", "path": str(path)})

    for o, path in zip(b.outputs, output_paths):
        n = int(np.prod(o.shape))
        buffers.append({
            "binding": int(o.binding), "role": "output",
            "path": str(path), "size": n * DTYPE_BYTES[o.dtype],
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


# --- chip bucketing ----------------------------------------------------------

_CHIP_TYPES = [
    ("M4 Max",   "m4_max"), ("M4 Pro",   "m4_pro"), ("M4",       "m4"),
    ("M3 Ultra", "m3_ultra"), ("M3 Max", "m3_max"), ("M3 Pro", "m3_pro"), ("M3", "m3"),
    ("M2 Ultra", "m2_ultra"), ("M2 Max", "m2_max"), ("M2 Pro", "m2_pro"), ("M2", "m2"),
    ("M1 Ultra", "m1_ultra"), ("M1 Max", "m1_max"), ("M1 Pro", "m1_pro"), ("M1", "m1"),
]


def chip_info() -> dict:
    """Detect the host chip. Mirrors src/metal_scripts/setup.cpp::detect_chip().

    Returns:
        {"type":"m2_max","name":"Apple M2 Max","bucket":"apple-m2-max",
         "cpu_cores":N,"gpu_cores":N,"ram_bytes":N}
    """
    name = "unknown"
    try:
        name = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        pass

    chip_type = "unknown"
    for needle, tag in _CHIP_TYPES:
        if needle in name:
            chip_type = tag
            break

    cpu_cores = 0
    ram_bytes = 0
    try:
        cpu_cores = int(subprocess.check_output(
            ["sysctl", "-n", "hw.physicalcpu"], text=True).strip())
    except Exception:
        pass
    try:
        ram_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True).strip())
    except Exception:
        pass

    gpu_cores = 0
    try:
        sp = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], text=True, timeout=5)
        import re
        m = re.search(r"Total Number of Cores:\s*(\d+)", sp)
        if m:
            gpu_cores = int(m.group(1))
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


def bucket_key_from_name(name: str) -> str:
    s = "".join(
        (c.lower() if c.isalnum() else "-") for c in name
    )
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-") or "unknown"


def bucket_key() -> str:
    """Canonical chip identifier for partitioning results.

    Derived from MTLDevice.name → sanitized lowercase with hyphens, e.g.
    "Apple M2 Max" → "apple-m2-max". Per-chip results live at
    `results/<bucket_key>/<kernel>.json`. GPU core count + RAM live inside
    the JSON for finer-grained sub-bucketing later (e.g. M2 Max 30c vs 38c).
    """
    return chip_info()["bucket"]


# --- MLX + system introspection ----------------------------------------------

def device_info() -> dict:
    """Identify the host bucket: chip + Metal device + memory."""
    info: dict = {
        "machine":   platform.machine(),
        "os":        f"{platform.system()} {platform.release()}",
    }
    for key, label in [
        ("machdep.cpu.brand_string", "cpu_brand"),
        ("hw.model",                 "hw_model"),
        ("hw.memsize",               "memory_bytes"),
    ]:
        try:
            info[label] = subprocess.check_output(
                ["sysctl", "-n", key], text=True
            ).strip()
        except Exception:
            pass

    try:
        info["mlx_metal_available"] = bool(mx.metal.is_available())
        info["mlx_device_info"]     = mx.metal.device_info()
    except Exception as e:
        info["mlx_error"] = repr(e)
    return info


@contextlib.contextmanager
def capture(path: str | Path) -> Iterator[Path]:
    """Record a Metal GPU trace of MLX dispatches.

    Wrap the timed loop in this to inspect what MLX is sending to the GPU —
    the resulting .gputrace opens directly in Xcode (Debug Navigator → Capture).
    Closest equivalent to `torch.compile`'s IR dump for built-in MLX ops.
    """
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    mx.metal.start_capture(str(p))
    try:
        yield p
    finally:
        mx.metal.stop_capture()


def array_summary(arr) -> dict:
    """Quick stats for sanity-checking an MLX or numpy array."""
    if isinstance(arr, mx.array):
        mx.eval(arr)
        np_arr = np.asarray(arr)
        dtype = str(arr.dtype)
    else:
        np_arr = np.asarray(arr)
        dtype = str(np_arr.dtype)
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


def mx_to_np_bytes(arr: mx.array) -> tuple[np.ndarray, str]:
    tag = MX_DTYPE_TAG.get(arr.dtype)
    if tag is None:
        raise ValueError(f"unsupported dtype {arr.dtype}; extend MX_DTYPE_TAG")
    mx.eval(arr)
    return np.asarray(arr, dtype=DTYPE_NP[tag]), tag
