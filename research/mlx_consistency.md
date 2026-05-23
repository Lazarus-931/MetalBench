# MLX Run-to-Run Consistency — Research Notes

Researched 2026-05-22 in response to the "Run-to-run speedup variance"
section of `mlx_findings.md`. The current `mlx/scripts/timing.py`
already follows the pattern shown in the MLX docs (warmup loop with
`mx.eval`, then timed loop with `mx.eval`, terminal `mx.synchronize`).
The remaining variance is not a barrier-correctness problem — it is a
GPU **performance state / DVFS** problem plus dispatch jitter for
sub-millisecond ops.

## Summary — three things to actually change

- **Anchor rankings to GPU-timestamp medians, not wall-clock speedup.**
  Apple explicitly states there is *no public API* to query or pin GPU
  clock on macOS — only Xcode's "Lock Performance State" dropdown
  during a GPU capture can do it. Since we cannot pin clocks from a
  script, stop treating MLX wall-clock as a stable baseline; use it
  for display only and rank with `cb.GPUStartTime/GPUEndTime` medians
  (already partly done in `harness.py`).
- **Drop "min_ms"; report median + a robust spread (e.g. p10/p90 or
  IQR) instead.** Apple-GPU min times are biased toward whatever
  microsecond happened to coincide with a high-frequency clock burst,
  which is exactly what swings between runs. Median over 200 iters
  with 50-iter warmup is reasonable; min is misleading.
- **For sub-ms ops, batch the work inside one dispatch before timing.**
  `awni` (MLX maintainer) and the MLX docs use `for _ in range(100):
  mx.eval(fun(x))` — but for kernels under ~200 µs the
  `time.perf_counter()` + Python loop overhead is itself ~10–30 µs and
  dispatch jitter dominates. Wrap N calls in a single graph
  (`y = fn(...); for _ in range(K): y = fn(y_or_x)`) and `mx.eval(y)`
  once, then divide — this is what `mlx-benchmark` (Tristan Bilot)
  effectively does to push the timed region above the noise floor.

## Concrete changes to `mlx/scripts/timing.py`

1. Add a `p10_ms`/`p90_ms` (or `iqr_ms`) field next to median. Stop
   relying on `min_ms` for ranking; keep it only for diagnostics.
2. Optionally add an `inner_repeats: int = 1` parameter. When the
   measured median falls under ~0.5 ms on the first pass, re-time
   with `inner_repeats` chosen so each timed step is ≥ 1 ms. Sum the
   ops into one graph and `mx.eval` once per outer iter. This trades
   memory for stability and is the standard fix for jittery
   microbenchmarks on Apple GPUs.
3. Keep `mx.eval` per-iter (correctness barrier). `mx.synchronize` is
   redundant inside the timed loop — `mx.eval` already blocks the
   calling thread until the GPU stream is drained — so do not add it
   per-iter. The trailing `mx.synchronize()` after warmup is fine.
4. Add a thermal-bias guard: record `time.monotonic()` before/after
   the whole `time_mlx` call and stash wall elapsed in the result;
   if a single timing took > some threshold (e.g. 5 s of wall for
   ~200 cheap iters), flag the result as "thermally suspect" so the
   harness can refuse to overwrite a prior best with it.
5. For ranking on amelia specifically, consider a short cooldown
   (`time.sleep(0.5)` between ops) and/or only accept new bests from
   lexie — this is a harness-level decision, not `timing.py`'s.

## Open questions

- Does MLX expose a public way to force a Metal command queue to
  high-priority / sustained mode? `MTLCommandQueue` has a private
  `MTLCommandQueuePriority` enum but it is not surfaced in MLX's
  Python API. Not confirmed.
- Whether `mx.metal.start_capture` (used for Xcode GPU traces) has any
  side effect on clock state during capture. Apple's perf-state lock
  is a *Xcode UI* feature tied to a capture session; whether it
  persists for a Python-driven workload outside Xcode is unclear from
  the docs.
- No published "official" MLX warmup count exists. The docs use 10;
  awni's discussion-1571 advice is just "use warmup + multiple
  measurements". 50 is defensible; >50 likely diminishing returns.
- `mx.compile`: enabling it would cut Python overhead but introduces
  its own first-call compile latency. Not yet measured here.

## Sources

- MLX docs, Compilation page (10-warmup, 100-iter pattern):
  https://ml-explore.github.io/mlx/build/html/usage/compile.html
- MLX discussion #1571 (mx.eval vs mx.async_eval, compile-on-first-call):
  https://github.com/ml-explore/mlx/discussions/1571
- MLX issue #2391 (CPU/GPU sync model overview):
  https://github.com/ml-explore/mlx/issues/2391
- Apple Developer Forums — "no public API to query GPU clock; lock
  performance state in Xcode":
  https://developer.apple.com/forums/thread/692062
- Apple — Optimizing GPU performance (perf-state dropdown):
  https://developer.apple.com/documentation/xcode/optimizing-gpu-performance
- Apple — Analyzing Apple GPU performance using counter statistics:
  https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics
- mlx-benchmark (Tristan Bilot) — community reference benchmark harness:
  https://github.com/TristanBilot/mlx-benchmark
- philipturner/metal-benchmarks — Apple GPU microarchitecture probes
  (saturation-based timing to dodge jitter):
  https://github.com/philipturner/metal-benchmarks
- arXiv 2510.18921 — "Benchmarking On-Device ML on Apple Silicon with
  MLX" (uses 5-iter averaging; no clock-locking mentioned):
  https://arxiv.org/abs/2510.18921
