# MetalBench

Agentic harness for authoring + benchmarking Apple Metal GPU kernels against [MLX](https://github.com/ml-explore/mlx) reference implementations. Modeled on [KernelBench](https://github.com/ScalingIntelligence/KernelBench), swapping CUDA → Metal and PyTorch → MLX.

## Layout

```
src/
  kernels/        # .metal sources — one kernel (or fused kernel) per file
  host/main.mm    # generic Objective-C++ runner: loads metallib, dispatches, times
build/            # compiled .air, .metallib, host binary (gitignored)
python/metalbench/
  task.py         # Task dataclass — defines a single benchmark problem
  host.py         # writes manifest, invokes host binary, reads outputs back
  eval.py         # runs kernel + MLX reference, compares + computes speedup
  cli.py          # `python -m metalbench.cli <task.py>`
tasks/            # task definitions, organized by level
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
make            # builds host binary + every src/kernels/*.metal
```

Requires Xcode command line tools (`xcrun -sdk macosx metal`) and an Apple Silicon (or Metal-capable) Mac.

## Authoring a task

A task is a Python file that exports `task: Task`. The author provides:

- a `.metal` kernel (compiled to `build/<name>.metallib` by `make`)
- `make_inputs(seed)` — returns input `mx.array`s in binding order
- `outputs` — what the host should allocate and read back
- `scalars_fn(inputs)` — scalar bindings (sizes, alphas, etc.)
- `grid_fn(inputs)` + `threadgroup` — launch config for `dispatchThreads:`
- `reference(*inputs)` — MLX reference implementation

## Running

```bash
metalbench tasks/level1/<task>.py --iters 100
```

Output is a JSON dict with `correct`, `speedup` (MLX median / kernel median), and per-output max error.

## Manifest contract (host ↔ Python)

The Python harness writes a JSON manifest describing one launch; the C++ host consumes it. See the docstring at the top of `src/host/main.mm` for the schema.
