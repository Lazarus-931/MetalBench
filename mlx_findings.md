# MLX Performance Analysis for MetalBench

## Method

Analyzed the [MLX source](https://github.com/ml-explore/mlx) (commit at
`/Users/alazarmanakelew/mlx/`) and disassembled both MLX's 125 MB
`mlx.metallib` and our `build/*.metallib` using `metal-objdump -d`.

---

## Tile Configuration

For float32 matrix multiply on M2 (`devc='g'`), MLX uses exactly:

```
BM=64, BN=64, BK=16, WM=2, WN=2
```

The `GEMM_TPARAM_MACRO` at `matmul.cpp:88` adjusts for device class:
- `devc == 'g' || devc == 'p'` (iPhone/iPad GPU, M2 base): keeps defaults
- `devc == 'd'` (M2 Pro/Max/Ultra): overrides for large matmuls

For float16, MLX uses much larger tiles (`BK=256`, `BK=512`) because half the
memory per element allows fitting larger K-slices in threadgroup memory.

For float32 with `nn` (no transpose A, no transpose B), the macro **does not
override** the defaults. MLX and MetalBench use identical tile sizes.

---

## Key Architectural Difference: Simdgroup Layout

| | MLX | MetalBench |
|---|---|---|
| Simdgroups | 4 (2×2, WM=2,WN=2) | 8 (4×2) |
| Threads/group | 128 | 256 |
| Accumulators/simdgroup | 16 (TM=4,TN=4) | 8 (TM=2,TN=4) |
| Total accumulators/group | 64 | 64 |
| Loop style | Single-buffered, 2 barriers/iter | Double-buffered, 1 barrier/iter |

Same total accumulators per threadgroup, different distribution.

---

## Critical Finding: float2 Accumulator Packing

Disassembling `mlx.metallib`:

```
; MLX accumulator type
%"struct.mlx::steel::MMATile.3" = type { [16 x <2 x float>] }
```

Disassembling our `build/rect_mm.metallib`:

```
; Our accumulator type
%"struct.metal::simdgroup_matrix" = type { <64 x float> }
```

**MLX stores each 8×8 accumulator as 32×float2 (32 registers).**
**We store each 8×8 accumulator as 64 individual floats (64 registers).**

This float2 packing is the result of MLX's C++ template wrappers
(`MMATile<T, TM, TN, MMAFrag>` with `frag_type = vec<float, 2>`).
The Metal compiler recognizes these and allocates registers accordingly.

### Why this matters

- MLX: 16 accumulators × 32 float2 = **512 registers** per simdgroup (16/thread)
- MetalBench: 8 accumulators × 64 float = **512 registers** per simdgroup (16/thread)

**Same register pressure, but MLX has 2× the instruction-level parallelism**
(16 independent MMA operations per barrier vs our 8). MLX's 4-simdgroup design
leaves half the M2 SIMD units idle but compensates with more ILP per active unit.

When we tried the 4-simdgroup/16-accumulator layout in raw MSL, the compiler
did NOT pack accumulators as float2 — it allocated 64 floats per accumulator,
doubling register pressure (32 floats/thread → spilling → 5× slowdown at 52ms
vs 9.6ms).

---

## Threadgroup Memory

Both MLX and MetalBench use identical padded memory:

```
As: BM × (BK + 4) = 64 × 20 = 1280 floats = 5 KB
Bs: BK × (BN + 4) = 16 × 68 = 1088 floats = 4.25 KB
Total (single-buffered): 9.25 KB
Total (double-buffered): 18.5 KB
```

Padding of 4 floats (16 bytes) per row avoids threadgroup memory bank
conflicts. Stride of 20 (BK+4) is not a power of 2, preventing column
accesses from all hitting the same memory bank.

---

## Dispatched Assembly Comparison

**MLX inner loop** (from `steel_gemm_fused_nn_float32_bm64_bn64_bk16_wm2_wn2`):

```
; Load A tile
air.simdgroup_matrix_8x8_load (stride 20, 8×8 from As)
air.simdgroup.barrier(mem_none)
; Load B tile  
air.simdgroup_matrix_8x8_load (stride 68, 8×8 from Bs)
air.simdgroup.barrier(mem_none)
; MMA accumulate
air.simdgroup_matrix_8x8_multiply_accumulate
; Advance K offset, repeat for kc=8
```

**MetalBench inner loop** (from `rect_matmul_f32`):

```
; Load A_blk[2] (kc=0 and kc=8)
air.simdgroup_matrix_8x8_load (stride 20, 8×8 from As)
; Load B_blk[4]
air.simdgroup_matrix_8x8_load (stride 68, 8×8 from Bs)
; MMA accumulate 8 times (2×4)
air.simdgroup_matrix_8x8_multiply_accumulate
```

Both use identical AIR intrinsics. The difference is in the accumulator
storage type and the number of accumulators per simdgroup.

---

## Device-Specific Tuning

From `matmul.cpp:88`, MLX's per-device tile selection:

```
Device 'g'/'p' (M1, M2, iPhone, iPad):
  float32 nn: bm=64, bn=64, bk=16, wm=2, wn=2 (defaults, not overridden)
  complex64:  bm=64, bn=32, bk=8,  wm=4, wn=1
  float16:    bm=64, bn=64, bk=16, wm=1, wn=2 (fewer simdgroups)
  nt case:    bm=64, bn=32, bk=32, wm=2, wn=2

Device 'd' (M2 Pro, M2 Max, M2 Ultra):
  Large matmuls (>1M elements):
    float16:  bm=64, bn=64, bk=16, wm=1, wn=2
    float32:  bm=128, bn=128, bk=512 (much larger tiles!)
  Small matmuls: same as device 'g' defaults
```

For rect_mm (1024×4096 × 4096×2048 = 17.2B FLOPs), M2 uses the small dev 'g'
path even though this is a large compute problem.

---

## Performance Ceiling on M2

| kernel | MetalBench | MLX | GFLOPS | % of M2 peak |
|---|---|---|---|---|
| sqr_mm | 1.13ms | 1.37ms | 1897 | 53% |
| rect_mm | 9.56ms | 7.90ms | 1797 | 50% |
| batch_mm | 2.76ms | 2.17ms | 1554 | 43% |

M2 theoretical peak: ~3.6 TFLOPS.

Our float32 matmul kernels plateau at ~1700-1900 GFLOPS (~50% of peak)
regardless of tile or loop variations. MLX reaches ~2200 GFLOPS (~61%).

The ~20% gap stems from float2 accumulator packing that MLX's C++ template
system achieves. Reproducing this in pure Metal Shading Language would require
either:

1. Direct `float2` vector intrinsics for accumulator storage (not supported
   by `simdgroup_matrix` API)
2. Inline AIR assembly (not supported in MSL)
3. A C++ template layer matching MLX's `MMATile` design

---

## What Worked

### Large speedups (element-wise kernels)
Float4 vectorized grid-stride loops saturate M2's ~89 GB/s memory bandwidth.
MLX dispatch overhead (~0.15ms per op) dominates for sub-millisecond GPU work.

### Simdgroup reductions (norms, scans)
`simd_sum`, `simd_prefix_inclusive_sum`, and `simd_shuffle_down` accelerate
reductions and prefix scans by 4-10× vs threadgroup-only approaches.

### Padded threadgroup memory
Adding 4-float padding per row to avoid power-of-2 stride bank conflicts
improved matmul throughput by 2-5%.

---

## What Didn't Work

| approach | result |
|---|---|
| Larger K-tiles (BK=32,64) | Regressed: more threadgroup memory → lower occupancy |
| Smaller K-tiles (BK=8) | Regressed: more barrier iterations |
| Larger tiles (128×64, 64×128, 128×128) | Regressed: register pressure |
| 4-simdgroup layout (WM=2,WN=2) | 52ms: register spill from unpacked accumulators |
| Single-buffered loop | Same perf as double-buffered |
| Manual inner loop unrolling | Regressed |
| `simdgroup_barrier` scheduling hints | Regressed or no change |
| 2×4 simdgroup layout | Slightly slower than 4×2 |

---

## Run-to-run speedup variance — diagnosis (2026-05-22)

Observed: re-running the same kernel produces speedups that swing
2–10× between runs (e.g. `relu` on M2 oscillating 30×, 80×, 150×).
The git history for `results/apple-m2/*.json` and `results/apple-m4/*.json`
shows this clearly across the recent refresh waves — many results
were re-recorded simply because *that run's MLX baseline happened to
be slow*, not because the kernel got faster.

Root causes, ranked by impact:

1. **Sub-ms kernels are jitter-dominated.** A 30 µs kernel against a
   7 ms MLX baseline produces a 233× speedup. A 30 µs kernel against
   the same op when MLX happens to JIT-trace faster (0.5 ms) is 16×.
   Same kernel, same hardware — the *baseline* moved.
2. **MLX warmup of 10 iters was insufficient.** `mx.eval` per iter is
   a correctness barrier, not a clock-stabilization barrier. Apple's
   GPU dynamic frequency scaling needs sustained sub-ms work to reach
   steady state. Bumped CLI default `--warmup` to 50
   (branch `mlx-variance-fix`).
3. **Thermal throttling on amelia.** Same kernel posts MLX baselines
   varying 100× depending on what was just running on the mini.
   Lexie is the cleanest (consistent), derek middle, amelia worst.
   For results.md "best speedup" purposes we should preferentially
   record on lexie.
4. **Per-iter `mx.eval()` was already in place.** `time_mlx` in
   `src/mlx_scripts/timing.py` line 57 has `mx.eval(out)` inside the
   timed loop — so the per-iter barrier the user asked for already
   exists. The variance came from #1–#3, not from missing barriers.

### Methodology fix going forward

- Bigger `--warmup` (50, done).
- For "best on chip X" rankings, use **kernel median_ms only**
  (chip-side GPU timestamps from `cb.GPUStartTime/GPUEndTime` in
  `src/metal_scripts/timing.mm`) — speedup vs MLX is a display
  metric, not a ranking metric. `update_session` already ranks by
  kernel median_ms (`src/mlx_scripts/harness.py` line ~199) so this
  is already partly correct; ensure the website's "best" column
  follows the same rule and doesn't surface speedup as canonical.
- Don't re-record results just because speedup got bigger — only
  when kernel median_ms drops.
