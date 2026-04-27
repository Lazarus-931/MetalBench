from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, Tuple

import mlx.core as mx

DType = str  # "f32" | "i32" | "u32"


@dataclass
class OutputSpec:
    binding: int
    dtype: DType
    shape: Sequence[int]


@dataclass
class ScalarSpec:
    binding: int
    dtype: DType
    value: float | int


@dataclass
class Task:
    """A single MetalBench problem.

    Authoring contract:
      - `metallib` points at a compiled .metallib (built by `make` from
        a .metal file in src/kernels/).
      - `function` is the kernel symbol inside that metallib.
      - `make_inputs(seed)` returns the input mx.arrays in the same order
        as `input_bindings` (i.e. inputs[i] is bound at input_bindings[i]).
      - `outputs` declares the output buffers the host should allocate and
        read back. The harness reconstructs mx.arrays of the given shape/dtype.
      - `scalars_fn(inputs)` returns scalar bindings (e.g. N, M, K, alpha).
      - `grid_fn(inputs)` returns the (x, y, z) grid for `dispatchThreads:`.
      - `threadgroup` is the threads-per-threadgroup tuple.
      - `reference(*inputs)` is the MLX reference. Returns one mx.array or a
        tuple matching `outputs`.
      - `rtol`/`atol` are the correctness thresholds.
    """
    name: str
    metallib: str
    function: str
    input_bindings: Sequence[int]
    outputs: Sequence[OutputSpec]
    grid_fn: Callable[[Sequence[mx.array]], Tuple[int, int, int]]
    threadgroup: Tuple[int, int, int]
    make_inputs: Callable[[int], Sequence[mx.array]]
    reference: Callable[..., Any]
    scalars_fn: Callable[[Sequence[mx.array]], Sequence[ScalarSpec]] = field(
        default=lambda _inputs: []
    )
    rtol: float = 1e-4
    atol: float = 1e-5
