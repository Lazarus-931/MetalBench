# Anti-patterns — hard rules for Metal kernel agents

This file is injected verbatim into the Implementor agent's system prompt
(via `prompts/implementor.md`). Every rule here came from a real failure
the project hit and resolved. Adding rules is encouraged when a new failure
class is identified and fixed; deleting rules requires evidence the failure
mode no longer applies (chip generation change, harness update, etc.).

## Dispatch geometry

1. **`threads_per_threadgroup > pso_max_threads_per_tg` silently fails on M-series.**
   The Metal driver caps the threadgroup at the PSO's reported max after
   compilation (M2 commonly returns 896 for register-heavy kernels even when
   the kernel header requests 1024). The dispatch *appears* to run — the
   harness still reports a kernel time — but the output buffer stays zero
   and correctness fails with `max_err=1.0`.
   - Bit `conv_transpose2d_sub_tanh` on M2 (1024 requested, 896 cap → fix
     was to drop tg to 512).
   - Check `BenchResult.max_threads_per_tg` before proposing any
     threadgroup change. Never exceed it.

2. **Threadgroup memory budget on Apple M-series is ~32 KB per group.**
   Adding `threadgroup` arrays that push total static TG memory beyond
   ~32,768 bytes makes PSO creation fail. The harness reports
   `tg_static_mem_bytes` — compute (current + your additions) before
   adding any `threadgroup float[N]`.

## Numerical / correctness

3. **fp16 accumulators on attention softmax produce out-of-tolerance error.**
   The score-mass after `exp(score - max)` summed across 64+ positions
   overflows fp16's representable range or drifts beyond `rtol=1e-2`.
   Use fp32 for the softmax-sum accumulator. fp16 is fine for the
   subsequent value-multiply if the V tile is fp16.

4. **Half-precision intermediate storage of multiplied feature maps**
   (`half tmp[H*W*C]` for conv intermediates) drifts beyond `rtol=1e-2`
   when channels > 32. Keep intermediates fp32 unless explicitly verified
   correct on the target shape.

5. **`fast::tanh` / `fast::exp` are acceptable for GELU activation but
   not for the softmax max-shift normalization** — the max-shift trick
   needs precise round-half-to-even behavior.

## Timing / measurement

6. **`kernel_ms < 0.001` is below timing resolution.** Speedup numbers
   become garbage (kernel returning ~0 means `mlx_ms / kernel_ms` blows
   up to thousands). Treat such results as "unbenched" — do not record,
   do not iterate.

7. **`mean_ms / median_ms > 1.5` is jitter.** The kernel is launch-
   overhead-dominated; micro-optimizations won't beat the noise floor.
   Structural changes (multi-TG dispatch, batch-up of small kernels) are
   the only viable path. Don't waste rounds on float4 or unroll on these.

8. **amelia mini is thermally noisy.** Same kernel runs 3.1 ms / 4.1 ms /
   5.2 ms between cold/warm/hot states. For amelia, require 5-run
   consistency at CV < 0.15; lexie and derek are stable enough at 3 runs.

## Cross-chip rule (MetalBench-specific)

9. **An M4 win is not a real win until M2 confirms.** Apply the proposed
   edit, bench on M4 (the originating chip), then bench on M2. Two
   outcomes:
   - Both improve (or M2 ties) → keep the file flat as
     `<name>.metal`. One impl for all chips.
   - M2 regresses by > 2% → split. Put your edit at
     `<name>/m4.metal`, restore the original to `<name>/default.metal`.
   The harness routes `<name>__<chip>.metallib` → `__default.metallib`
   automatically.

## What is OK and what is not

10. **Editing `mlx/kernels/<set>/<name>.py` (the MLX baseline) is always
    rejected.** It defines the problem. If your technique requires
    changing the MLX baseline, your technique is wrong.

11. **Changing `input_bindings`, `output_shape`, `metal_function`,
    `flops`, or `bytes` in `registry.py` is rejected.** Those are spec.
    `grid` and `threadgroup` are the only registry fields you may
    modify, and only when a dispatch-shape change is the technique
    itself (e.g. bias_add 64k → 8k grid for TG-cache amortization).

12. **One technique per diff.** Don't bundle multiple changes; the
    Verifier can't disambiguate which one moved the number.
