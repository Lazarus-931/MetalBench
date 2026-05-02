# Contributing to MetalBench

MetalBench is an open benchmark for Apple Metal GPU kernels. Submit a kernel
that beats the current best time and it becomes the new reference! simple as that!

## Do this

```bash
git clone https://github.com/Lazarus-931/MetalBench.git
cd MetalBench
python3 setup.py                      # one-time: installs toolchain + deps
./bench sqr_mm                        # build + run the square matmul kernel
```

## Submitting a kernel

1. **Fork** the repo.
2. **Pick a kernel** from [best_times.md](best_times.md) you want to improve, or add a new one.
3. **Edit** `src/kernels/common/<name>.metal` — change the implementation, not the function signature, that will break code!
4. **Run** `./bench <name>` until `correct=true` and you have a new best time.
5. **Update** `best_times.md` with your new time and speedup.
6. **PR** with only:
   - The `.metal` file you changed
   - Updated `best_times.md` with the actual time


## Adding a new kernel

1. Create `mlx/kernels/common/<name>.py` with just a `Model(nn.Module)` class and its `forward()`.
2. Create `src/kernels/common/<name>.metal` with your Metal kernel.
3. Add a registry entry in `mlx/kernels/common/registry.py` with `metal_function`, bindings, grid, shapes.
4. Run `./bench <name>` to verify correctness.
5. Add your entry to `best_times.md`.

## Scoring

Every benchmark prints all 5 targets:

| target | what it measures | good means |
|---|---|---|
| `speed` | speedup vs MLX | > 1.0 = faster than MLX |
| `compute` | GFLOPS | higher = better GPU utilization |
| `memory` | GB/s | near M2 peak (~89 GB/s) |
| `stable` | consistency (0–1) | > 0.95 |
| `balanced` | weighted composite | higher = better overall |

## NetalBench structure

```
MetalBench/
├── bench                          # CLI entry point
├── best_times.md                  # current leaderboard
├── setup.py                       # one-time environment check
├── Makefile                       # builds .metal → .metallib + host binary
├── src/
│   ├── kernels/common/            # your Metal kernels go here
│   │   └── utils/utils.metal      # shared Metal helpers
│   ├── metal_scripts/             # host binary (C++/ObjC)
│   └── mlx_scripts/               # Python harness
├── mlx/kernels/common/            # MLX baselines (Model class only)
│   └── registry.py                # kernel dispatch metadata
├── results/<chip>/                # per-chip benchmark results
└── session.json                   # best times per kernel per chip
```

## Need help?

Open an issue. Tag it `kernel-idea` for new kernel proposals or `help-wanted` for optimization advice.
