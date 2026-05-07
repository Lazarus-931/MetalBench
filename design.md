# MetalBench Design

## Architecture

```
./bench <name>                     # CLI entry point
  → harness.py                     # orchestrates build + run + grade
    → mlx_helpers.py               # loads Model from .py + metadata from registry
    → main.mm (metalbench_host)    # compiles .metal → dispatches → times via GPU timestamps
    → timing.py                    # times MLX reference via mx.eval()
```

### Separation of concerns

| layer | file(s) | what it owns |
|---|---|---|
| **MLX reference** | `mlx/kernels/<set>/<name>.py` | `Model(nn.Module)` class only. Defines the operation. |
| **Kernel metadata** | `<set>/registry.py` | `metal_function`, bindings, grid, scalars, flops, bytes. One dict per kernel. |
| **Metal kernel** | `src/kernels/<set>/<name>.metal` | GPU implementation. Edited by contributors. |
| **Harness** | `src/mlx_scripts/` | Never touched by contributors. Auto-generates get_inputs, make_inputs, reference from Model + registry. |
| **Host binary** | `src/metal_scripts/` | C++/ObjC. Compiles metallib, dispatches, times via GPU timestamps. |

### Why registry over per-file metadata

Early versions had each `.py` file contain its own `metal_function`, `threadgroup`, `grid`, `scalars`, etc. This meant:
- Changing dispatch params required editing the baseline (which defines the problem)
- Boilerplate was copy-pasted across 30+ files
- No single place to audit all kernels

The registry centralizes all dispatch metadata. Each `.py` file is purely the MLX reference — ~8 lines of Model class. The harness auto-generates everything else.

### Why GPU timestamps for Metal, wall clock for MLX

Metal kernels are timed via `MTLCommandBuffer.GPUEndTime - GPUStartTime` — pure GPU execution time, no OS noise. MLX is timed via `time.perf_counter()` around `mx.eval()` — wall clock including dispatch overhead.

This means the speedup metric slightly favors Metal (GPU time vs wall clock). For element-wise kernels where dispatch overhead dominates (~0.15ms), this creates the 10-15× speedups we see. For matmuls (1-10ms GPU time), dispatch overhead is negligible and speedups are 0.8-1.4×.

### Tile design

All float32 matmul kernels use the same tile after extensive search:

```
BM=64, BN=64, BK=16          tile dimensions
SM=16, SN=32                 simdgroup layout (4×2 = 8 simdgroups)
MMA_M=2, MMA_N=4             8 accumulators per simdgroup
256 threads                  8 simdgroups × 32 threads
```

Double-buffered with padded threadgroup memory (LDA=BK+4, LDB=BN+4). This is the sweet spot on M2 — larger tiles cause register spilling, smaller tiles have too many barriers.

### Multi-chip support

Chips are auto-detected via `machdep.cpu.brand_string` on macOS. Results are bucketed per chip in `results/<bucket>/`. `tuning.py` allows per-chip dispatch parameter overrides without touching kernels. The harness checks for `(name, chip_type)` and falls back to `(name, chip_generation)` then registry defaults.

### Submission workflow

1. Fork → edit `.metal` → `./bench <name>` → update `best_times.md` → PR
2. Only `.metal` files changed. No harness touched. No baseline touched.
3. `session.json` stores the winning kernel source for reproducibility.
