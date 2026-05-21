# AGENTS.md

This is for automated agents (Claude Code, etc.) contributing Metal kernels.

## Your job

Make `./bench <name>` come back with `correct=true` and the highest possible
`speedup` against the MLX reference. Then update `best_times.md`.

## How a benchmark is structured

Each kernel has three files, paired by name:

| file | who edits it | role |
|---|---|---|
| `mlx/kernels/common/<name>.py` | project owner | **Model class only** — the MLX reference implementation |
| `mlx/kernels/common/registry.py` | project owner | dispatch metadata (metal_function, threadgroup, grid, scalars, flops, bytes) |
| `src/kernels/common/<name>.metal` | **you** | the Metal kernel you're optimizing |
| `src/kernels/common/<name>/<chip>.metal` | **you** | chip-specific variant (optional); `<chip>` ∈ {`default`, `m1`, `m2`, `m3`, `m4`, `m5`} |

The harness auto-generates `get_inputs`, `make_inputs`, `reference` from the
Model class + registry entry. You never touch the harness.

## Workflow

1. **First time:** `python3 setup.py` — installs Metal toolchain + Python deps + builds host.
2. **Read the baseline** at `mlx/kernels/common/<name>.py`. Just the `Model.forward()` tells you the operation.
3. **Read the registry entry** at `mlx/kernels/common/registry.py` for the kernel's `metal_function`, binding indices, grid, and scalars.
4. **Write/edit** `src/kernels/common/<name>.metal`. The kernel function name must match `metal_function`. Buffer bindings must match `input_bindings` and registry scalars.
5. **Run** `./bench <name>`. Checks correctness, prints all 5 target scores.
6. **Update** `best_times.md` with your new time + speedup.
7. **Open a PR** with only the `.metal` file changed + updated `best_times.md`.

## Per-chip variants (optional)

Most kernels ship as a single `src/kernels/<set>/<name>.metal` used on every
M-series chip. When a kernel genuinely needs different impls per generation
(e.g. M4 tensor cores, M5 new SIMD ops), promote it to a directory:

```
src/kernels/common/sqr_mm/
    default.metal    # fallback for any chip without its own file
    m4.metal         # M4-specific impl
    m5.metal         # M5-specific impl
```

Selection at bench time: `<name>__<chip>.metallib` → `<name>__default.metallib`
→ flat `<name>.metallib`. Don't promote until you have a measured perf reason —
the flat-file pattern is the default for the ~80% of kernels that don't need
chip-specific code.

## Rules

- **Don't edit baselines.** `mlx/kernels/common/<name>.py` is the spec.
- **Don't edit the harness.** `src/mlx_scripts/`, `src/metal_scripts/`, `Makefile`, `bench` are infrastructure.
- **Don't edit registry.py** unless adding a NEW kernel (not optimizing an existing one).
- **One kernel per PR.** Keeps review simple.
- **Don't claim unreproducible numbers.** Numbers come from `./bench`, never made up.

## What "fast" means

- `speedup` = MLX median / kernel median. >1.0 = beating MLX.
- All 5 targets printed on every run. Pick the right one for your kernel:
  - Element-wise → look at `memory` (GB/s)
  - Matmul → look at `compute` (GFLOPS)
  - Reductions → look at `speed`
- Results per-chip in `results/<bucket>/<name>.json`. M2 tuning may not transfer to M4.

## Optimization recipes

| op family | first try | next try |
|---|---|---|
| element-wise | float4 grid-stride loop, 1024 thr/tg | 64K threads for memory saturation |
| reductions | simd_sum + cross-simdgroup shuffle | fused multiple reductions in one pass |
| matmul | 64×64 tile, BK=16, 256 thr, double-buffered, padded tg mem | (ceiling ~50% peak on M2 in pure MSL) |
| norms (layer/rms) | simd_sum reduction per row, 1024 thr/tg | float2 accumulators |
| scans (cumsum) | `simd_prefix_inclusive_sum` + 2-level | hardware prefix ops beat manual |
