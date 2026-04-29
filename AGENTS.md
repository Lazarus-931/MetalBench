# AGENTS.md

Read this once before doing anything in this repo. Both human and LLM agents (Claude Code, Codex, etc.) should follow it.

## Your job

Make `./bench <name>` come back with `correct=true` and the highest possible `speedup` against the MLX reference. That's it.

## How a benchmark is structured

Each kernel slot has exactly two files, paired by name:

| file | who edits it | role |
|---|---|---|
| `mlx/kernels/<set>/<name>.py`   | the project owner | MLX baseline + correctness reference + the metal-side contract (function name, threadgroup, bindings, outputs) |
| `src/kernels/<set>/<name>.metal` | **you, the agent**  | the Metal kernel you're optimizing |

`<set>` is one of `common`, `standard`, `full`. Names are abbreviations like `sqr_mm`, `rect_mm`, `batch_mm`, `softmax_1d` — short, snake_case, no leading slot numbers. The same `<name>` is used in both file paths and as the metallib stem (`build/<name>.metallib`).

Shared Metal helpers live in `src/kernels/utils/utils.metal`. Add to it only when 2+ kernels need the same thing.

## Workflow

1. **First time only:** `python3 setup.py` — installs the Metal toolchain and Python deps, builds the host. Takes a few minutes the first time, instant after.
2. **Find your slot** in [KERNELS.md](KERNELS.md). Read the row.
3. **Read the baseline** at `mlx/kernels/<set>/<name>.py`. It tells you:
   - the kernel function name to expose (`metal_function`)
   - which buffers are inputs/outputs/scalars (`input_bindings`, `outputs`, `scalars`)
   - the threadgroup size and grid (`threadgroup`, `grid`)
   - the correctness tolerance (`rtol`, `atol`)
4. **Read at least one existing kernel** (e.g. `src/kernels/common/sqr_mm.metal`) to see the manifest contract in action.
5. **Write/edit `src/kernels/<set>/<name>.metal`.** Function signature must match the baseline's `metal_function` and binding indices.
6. **Run `./bench <name>`.** Output prints `[chip] ... ` first (the result bucket), then the JSON. Exit 0 = correct.
7. **Iterate.** If wrong: read the per-output `max_err` and use `python3 src/mlx_scripts/diff_arrays.py` for index-level diffs. If slow: add tiling, threadgroup memory, simdgroup_matrix.

## Rules — don't break these

- **Don't edit baselines.** `mlx/kernels/**/*.py` is the spec. Changing it is changing the benchmark.
- **Don't edit the harness.** `src/mlx_scripts/`, `src/metal_scripts/`, `Makefile`, `bench` are infrastructure. Bug? Open an issue.
- **Don't add Python or system deps.** If you think you need one, you probably don't.
- **Don't rename files or change conventions.** Same name across `.py` and `.metal`.
- **Don't claim a result you can't reproduce.** Numbers come from `./bench`, never made up.

## What "fast" means here

- `speedup` in the result JSON = `mlx_median_ms / kernel_median_ms`. Above 1.0 means you beat MLX's reference. Below means MLX is still winning.
- Results are partitioned per-chip in `results/<chip-bucket>/<name>.json` — what's fast on M2 may not be on M4. Don't tune to a number from a different chip.
- MLX's matmul / attention / norm kernels are *highly* optimized (often hardware MMA). Beating them takes care; matching them is already a win.

## Known optimization recipes (start here)

| op family | first try | next try |
|---|---|---|
| element-wise (abs, exp, …) | one thread per element, 256-wide threadgroups | vectorized loads (`float4`) |
| reductions (sum, mean, …)  | threadgroup reduction via `tg_sum` from `utils.metal` | warp/simdgroup reduce + threadgroup combine |
| matmul                     | 16×16 threadgroup tiling                  | `metal::simdgroup_matrix<float, 8, 8>` MMA, 32×32 output tiles |
| layer norm / rms norm      | one block per row, threadgroup reduction  | fused stats + normalize, vectorized stores |

## Verifying without a full bench

- `python3 src/mlx_scripts/harness.py <name> --dry-run` — builds the manifest and prints it without dispatching. Use this to check binding indices, dtype, grid.
- `make kernels` — compile your `.metal` files only. Catches syntax errors fast.

## Where to put a result

Always run with `--save` (the default in `./bench`). It writes `results/<chip-bucket>/<name>.json`. That's the artifact. If your run doesn't save, it didn't happen.
