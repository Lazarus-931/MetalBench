<p align="center">
  <img src="assets/logo.png" alt="MetalBench" width="240">
</p>

# MetalBench

Agentic harness for authoring + benchmarking Apple Metal GPU kernels against [MLX](https://github.com/ml-explore/mlx) reference implementations. Modeled on [KernelBench](https://github.com/ScalingIntelligence/KernelBench), swapping CUDA → Metal and PyTorch → MLX.


While working on SuperKittens, I found a agent based loop of kernel writing & testing against a written test for perf/acc helped alot along the way. This contains harneseses and code used to benchmark against baseline mlx versions. Kernels don't differ much, just how threadgroups are utilized. One of the main differences was performance across a array of newer m series chips. I soon realized this was a we-have-kernel-bench-at-home version I made, so I polished some stuff and releasing this as a benchmark + as a repo to continue your own metal kernel testing using any model. Much of this repo is orgaized/inspired by KernelBench, so props for them!


## Kernels
I've decided to split it into 4 types:

* Common Set - 100 of the most used operations & bits of a normal kernel, stuff such as matrix multipication in it's most basic shape, convolutions & layernorm to name a few
* Standard Set - 50 of most semi-difficult, common fused kernels, which brings more variations & use of memory and compute into play
* Full Set - 25 of most large scale, multiple operation kernels, similar to full steps, and can span entire model architectures

## Evaluation

So the actual benchmark measures 2 things, accuracy + performance. Accuracy is the error difference from mlx implementation to 

## Setup

Two commands from a fresh clone to your first benchmark:

```bash
python3 setup.py           # checks toolchains, installs Metal toolchain + Python deps, builds host
./bench sqr_mm             # build kernel, run, save, print report
```

`setup.py` checks (and tries to fix) all of:

1. macOS + Apple Silicon
2. Xcode developer tools (`xcode-select -p`)
3. **Metal toolchain** — runs `xcodebuild -downloadComponent MetalToolchain` if missing (a few hundred MB; the usual blocker on a fresh Mac)
4. Python dependencies (`mlx`, `numpy`, `pydantic`)
5. Host binary builds (`make host`)
6. Chip detection works

If any step fails it tells you exactly what to run.

## Running a benchmark

```bash
./bench <name>                              # default: warmup=10, iters=300, target=speed
./bench <name> --no-save                    # don't write results/<chip>/<name>.json
./bench <name> -- --target compute --iters 500
./bench <name> -- --cold-start              # measure first-launch latency
```

Defaults are set so a single command gives a stable, publishable number. Bump `--iters` for tighter measurement.

## What you get back

A human-readable report on the terminal:

```
  sqr_mm    target=speed   score=1.428
  device      : Apple M2 (m2)  8 CPU / 8 GPU / 9 GB
  occupancy   : tg_mem=16KB  max_thr/tg=896
  correctness : ✓ correct     max_err=0.000e+00
  speedup     : 1.43× vs MLX
  kernel      : 1.266 ms  (min 1.194, mean 1.289, n=300)
  mlx ref     : 1.808 ms
  compute     : 1695.8 GFLOPS
  bandwidth   :    9.9 GB/s
  arith int.  :  170.7 FLOPs/byte
  stability   : 0.98

      target        score
  ----------   ----------
       speed        1.428
     compute     1695.83
      memory        9.94
      stable        0.98
    balanced        1.23
```

Plus three artifacts:

- **`results/<chip-bucket>/<name>.json`** — full result, every run (with `--save`, default on)
- **`session.json`** — per-chip leaderboard. Auto-updated when a run beats the recorded best for that kernel; stores the entire `.metal` source of the winning version so the result is reproducible from the file alone
- **stderr** — `updated session.json [apple-m2/sqr_mm]: new best 1.266 ms (was 1.808 ms, Δ +0.542)` when a run wins

## Grading targets

Every run computes all five scores and prints them in a table. The `--target` flag only changes which one becomes the headline `score`.

| target | metric | good | bad | what it means |
|---|---|---|---|---|
| **speed** | `speedup` vs MLX | > 1.0× = faster than MLX | < 1.0× = slower than MLX | How your kernel compares to Apple's reference |
| **compute** | GFLOPS | higher = better throughput | low = GPU underutilized | Raw compute. Ignore for memory-bound kernels (element-wise ops, copies) |
| **memory** | GB/s | near M2 peak ~89 GB/s | well below peak | Memory bandwidth utilization. The primary metric for element-wise kernels |
| **stable** | 0–1 | > 0.95 = solid | < 0.90 = noisy | Run-to-run consistency. Low stability means thermal throttling or OS interference |
| **balanced** | composite | higher = better overall | — | `0.5·speedup + 0.3·gflops/1000 + 0.2·stability` |

**Which target matters for your kernel:**

- **Element-wise ops** (relu, sigmoid, add, etc.): look at `memory` — they're bandwidth-bound, GFLOPS will be low by nature
- **Matmuls**: look at `compute` — they're compute-bound, GB/s will be low by nature  
- **Reductions** (layernorm, softmax, dot product): look at `speed` — they're latency-sensitive, mixed compute/bandwidth
- **Scans** (cumsum): look at `speed` or `balanced`

### Example: good vs bad scores

```
GOOD (sqr_mm):                      BAD (naive kernel):
      target        score                  target        score
  ----------   ----------              ----------   ----------
       speed        1.428                   speed        0.120
     compute     1695.83                  compute      142.50
      memory        9.94                   memory        0.83
      stable        0.98                   stable        0.52
    balanced        1.23                 balanced        0.17
```

## Authoring a kernel

See [AGENTS.md](AGENTS.md) for the full contract. Short version:

- `mlx/kernels/<set>/<name>.py` — the MLX baseline (don't edit; it defines the problem).
- `src/kernels/<set>/<name>.metal` — your kernel.
- Run `./bench <name>` until `correct=true`.
- Edit only the `.metal` file. Update `best_times.md` with your result. Open a PR.
