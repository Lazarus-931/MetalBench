<p align="center">
  <img src="assets/logo.png" alt="MetalBench" width="240">
</p>

# MetalBench - Metal-based kernel benchmark, library, and closed-loop agent authoring system 

Benchmarking Apple Metal GPU kernels against [MLX](https://github.com/ml-explore/mlx) reference implementations. Modeled on [KernelBench](https://github.com/ScalingIntelligence/KernelBench), swapping CUDA → Metal and PyTorch → MLX.

While working on inference for Apple Silicon, I found an agent-based loop of kernel writing & testing against a perf/accuracy test helped a lot along the way. This repo contains the harness and code used to benchmark against baseline MLX versions. Kernels don't differ much — mostly in how threadgroups are utilized. One of the main differences was performance across the newer M-series chips. I soon realized this was a we-have-kernel-bench-at-home version, so I polished it and am releasing it as a benchmark + agent harness for Metal kernel authoring (Agent Steel). Much of this repo is organized after / inspired by KernelBench, so props to them!

Feel free to contribute with kernels for other mchip types, I only had my hands on a m4 mini and m2.

## Agent-Steel 👨‍🏭

`agent_steel/` is an LLM-driven closed-loop harness that profiles a kernel,
writes a new candidate, and verifies the change against `./bench` — repeating
until it can't improve further. New kernels are bookended by a fourth agent
that authors the MLX/registry/Metal scaffolding and polishes the result for PR.

* **Welder** - New-kernel mode only. Authors `mlx/kernels/<set>/<name>.py` + appends the registry entry + writes the first `.metal`. Two-stage accuracy gate (Metal vs MLX, MLX vs an external PyTorch reference). Invoked again after the perf loop to polish for PR.
* **Profiler** - Parses the `.gputrace` + bench output, runs the chip-aware synthesizer (`chip_metrics/m{N}.py`), emits a 2-3 paragraph diagnosis.
* **Optimizer** - Reads the diagnosis + the kernel's AttemptDB log + current `.metal` + MLX reference, writes the next `.metal` in-place to `metal/kernels/<set>/<kernel>/<chip>.metal`, runs an accuracy gate (`./bench` correctness ≥ 99%) — retries up to 4× on accuracy fail.
* **Verifier** - Benches the promoted kernel (`--warmup 30 --iters 100`), compares mean vs `session.json`'s leaderboard best, logs ±Δ% to AttemptDB, reverts on regression.

AttemptDB is one JSONL per `(kernel, chip)` at `.agent-steel/history/`. Every
entry carries the `.metal` source snapshot + the chip-aware metrics dict from
that profile. `AttemptDB.top_n_by_time(kernel, chip, n)` returns the N fastest
kept attempts.

Run it:

```bash
# optimize an existing kernel
python -m agent_steel --kernel-name relu --loop --max-rounds 5
python -m agent_steel --kernel-name relu,softmax --parallel 2 --loop

# author a brand-new kernel (Welder), then chain into the perf loop
python -m agent_steel --welder my_kernel --description '...' --reference ref.py --loop
```

Concurrent agent-steel instances are safe: a process-wide flock serializes
`./bench` (GPU is one resource); a per-(kernel, chip) flock prevents two
processes from racing on the same kernel's lineage.

<img height=700 src="assets/figma_viz.png"/>

### Results — 5 standard kernels (30 rounds, M2)

Sample run on a slice of the standard set. Each curve is best-so-far perf gain vs the leaderboard baseline as the loop iterates; legend shows the final % beat.

<img src="assets/agent_steel_top5_standard.png" alt="Agent Steel — 5 standard kernels (30 rounds, M2)"/>

## Kernels

Three sets, increasing in size/complexity:

| set | what |
|---|---|
| **Common** | basic ops — activations, matmuls, norms, convs, scans |
| **Standard** | fused 2+ op kernels — attention, SwiGLU, RMSNorm + linear |
| **Full** | end-to-end model blocks — transformer block, AlexNet, ResNet, LLaMA decoder, DenseNet |

See `KERNELS.md` for the full registry.

Kernels can be split per M-chip generation when one impl genuinely needs different code than another (e.g. `<name>/default.metal` + `<name>/m4.metal`). The harness auto-picks the right variant at runtime based on the chip detected via `sysctl`. See [Per-chip variants](#per-chip-variants) below.

## Evaluation

The benchmark measures accuracy against the MLX reference and performance across five targets: speedup vs MLX, compute throughput (GFLOPS), memory bandwidth (GB/s), run-to-run stability (0-1), and a balanced composite score. Every kernel that passes correctness gets a row in `best_times.md`. 

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
./bench <name>                              # default: both MLX + Metal, paired
./bench <name> --mlx                        # MLX only
./bench <name> --metal                      # Metal only
./bench <name> --no-session                  # don't write results/<chip>/<name>.json
./bench <name> -- --target compute --iters 500
./bench <name> -- --cold-start              # measure first-launch latency
./bench --all                               # run every kernel in the registry
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

## Authoring a kernel(Will be moved into AgentSteel)

For Agent Steel internals see [agent_steel/README.md](agent_steel/README.md). Short version for human contributors:

- `mlx/kernels/<set>/<name>.py` — the MLX baseline (don't edit; it defines the problem).
- `metal/kernels/<set>/<name>.metal` — your kernel.
- Run `./bench <name>` until `correct=true`.
- Edit only the `.metal` file. Open a PR. `best_times.md` and `LINK.md` are auto-generated from `session.json`.

### Per-chip variants

Most kernels stay as a single flat file. When a kernel genuinely needs different
code per M-generation, promote it to a directory:

```
metal/kernels/common/sqr_mm/
    default.metal    # fallback (used by chips without their own variant)
    m4.metal         # M4-specific impl
```

The harness auto-picks `<name>__<chip>.metallib` → `__default` → flat `<name>.metallib`
based on the chip you're running on. Only split when you have a measured perf reason.

## Citation

If you use MetalBench in published work, please cite it as:

```bibtex
@misc{metalbench2026,
  title  = {MetalBench: Apple Metal GPU Kernel Benchmarks},
  author = {Manakelew, Alazar},
  year   = {2026},
  url    = {https://github.com/Lazarus-931/MetalBench},
  note   = {Live leaderboard: https://lazarus-931.github.io/leaderboard.html}
}
```
