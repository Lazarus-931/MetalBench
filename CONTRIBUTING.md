# Contributing to MetalBench

Submit a kernel that beats the current best time and it becomes the new reference.
A PR can change one kernel or many — both are fine.

## Quick start

```bash
git clone https://github.com/Lazarus-931/MetalBench.git
cd MetalBench
python3 setup.py        # one-time: installs Metal toolchain + Python deps
./bench sqr_mm          # smoke test
```

## Submitting a PR

1. **Fork** and create a branch.
2. **Edit** `metal/kernels/<set>/<name>.metal` (or add a per-chip variant — see below).
   Do not touch `mlx/kernels/...` (the baselines define the spec), `registry.py`, the harness, or the Makefile.
3. **Certify your changes:**
   ```bash
   ./certify              # benches every kernel you changed
   ```
   This runs `./bench <name>` for each modified kernel, confirms `correctness : ✓ correct`, captures the median time, and updates `session.json` with the new best. It also prints a copy-pasteable block for your PR description.
4. **Commit** the changed `.metal` file(s) and `registry.py` if you needed a dispatch-shape change. `best_times.md` and `LINK.md` are regenerated automatically from `session.json` — don't hand-edit them.
5. **Open the PR.** Title format suggestion: `<kernel>: <old>× → <new>× on <chip>` (or list multiple if the PR is broader).

## What a reviewer does

Any reviewer with an Apple Silicon Mac can verify your claims:

```bash
./verify 47             # PR number on the origin repo
./verify user/MetalBench:branch
```

`verify` checks out the PR, finds every `.metal` you changed, benches each in the PR's worktree 3× (median), compares against the times recorded in `session.json`, and prints a PASS/FAIL table. Tolerance is **±15%** on median time.

A PR is mergeable when at least one reviewer on the same chip family runs `./verify` and gets all green.

## Per-chip variants

Most kernels stay as a single flat `metal/kernels/<set>/<name>.metal` used on every M-series chip. If a kernel genuinely needs different code per generation, promote it to a directory:

```
metal/kernels/common/sqr_mm/
    default.metal    # used by any chip without its own file
    m4.metal         # used on M4 (any variant: base, Pro, Max, Ultra)
    m5.metal         # used on M5
```

**Resolver rule.** At bench time the harness picks `<kernel>/<chip>.metal` if it exists, else `<kernel>/default.metal`, else the flat `<kernel>.metal`. This means:
- M1 and M2 share `default.metal` until one of them gets its own file.
- M3 with its own `m3.metal` does not affect what M1/M2/M4 use.
- A chip-specific file is *additive* — adding `m4.metal` never changes what runs on any other chip.

**The fork rule.** Once you have a measured-win per-chip optimization:
- If `default.metal` already exists in the directory → just add `<chip>.metal` next to it.
- If only flat `<kernel>.metal` exists → promote: `mv <kernel>.metal <kernel>/default.metal`, then add your `<chip>.metal`.

**Agent Steel does this automatically.** Running agent-steel on a chip X always writes to `<kernel>/<chip>.metal` (creating the directory structure if needed). The shared `default.metal` is preserved untouched, so other chips never silently regress from another chip's session.

Don't promote unless you can measure a speedup that justifies the fork.

## Adding a brand-new kernel

If your PR adds a kernel that doesn't exist yet:

1. Add `mlx/kernels/<set>/<name>.py` — a single `class Model(nn.Module)` with `forward()`.
2. Add a registry entry in `mlx/kernels/<set>/registry.py` (`metal_function`, `input_shapes`, `output_shape`, `threadgroup`, `grid`, `scalars`).
3. Write `metal/kernels/<set>/<name>.metal`.
4. Add a row to `KERNELS.md`.
5. Then follow steps 3–5 above.

## Layout

```
MetalBench/
├── bench                            # CLI: build + run + grade
├── certify                          # author preflight (before PR)
├── verify                           # reviewer check (against a PR)
├── metal/kernels/<set>/<name>.metal   # your kernel goes here
├── mlx/kernels/<set>/<name>.py      # the MLX baseline (don't edit)
├── mlx/kernels/<set>/registry.py    # dispatch metadata
└── session.json                     # leaderboard source of truth (per-chip best times + winning sources)
```

## Scoring

Every `./bench` run prints five targets:

| target | what | good |
|---|---|---|
| `speed` | speedup vs MLX | > 1.0× |
| `compute` | GFLOPS | high on matmul-bound kernels |
| `memory` | GB/s | near chip peak on memory-bound kernels |
| `stable` | run-to-run consistency (0–1) | > 0.95 |
| `balanced` | composite | higher is better |

The leaderboard ranks on `speed` by default.

## Need help?

Open an issue. Tag `kernel-idea` for new kernel proposals, `help-wanted` for optimization advice.
