# Diagnose a kernel's bottleneck — analysis prompt template

Used to get a 2-line "what's limiting this kernel" hypothesis from an LLM.
Caller fills in:
- `{{kernel_name}}`
- `{{chip}}` — e.g. `m2`, `m4`
- `{{speedup}}`, `{{gflops}}`, `{{gbps}}`, `{{stability}}`, `{{kernel_ms}}`, `{{mlx_ms}}`
- `{{intensity}}`, `{{ridge}}`, `{{classification}}`
- `{{metal_source}}` — the full .metal file content

---

Diagnose what's limiting `{{kernel_name}}` on Apple {{chip}}.

## Measurements
- Kernel time: {{kernel_ms}} ms (MLX ref: {{mlx_ms}} ms — speedup {{speedup}}×)
- Compute: {{gflops}} GFLOPS · Bandwidth: {{gbps}} GB/s · Stability: {{stability}}
- Arithmetic intensity: {{intensity}} FLOPs/byte (ridge point: {{ridge}})
- Roofline classification: {{classification}}

## Current Metal source
```metal
{{metal_source}}
```

## Required output
Two short sentences:
1. One sentence naming the single most likely bottleneck.
2. One sentence proposing the next change to try, naming specific Metal idioms (float4, simd_sum, simdgroup_matrix MMA tile, threadgroup memory caching, etc.).

Do NOT write code. Do NOT enumerate every possible optimization. Pick one.
