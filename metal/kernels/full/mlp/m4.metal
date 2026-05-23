// mlp on M4: simdgroup_matrix MMA, fused layer1->GELU->layer2 across D2 chunks.
// Shapes: x(16,128) @ W1(128,512) -> GELU -> @ W2(512,128) -> GELU -> @ W3(128,10) -> y(16,10)
//
// v3 wins over v1:
//   - Per-chunk GELU on layer1 output applied via thread_elements() in registers
//     (eliminates a 2048-thread SMEM GELU pass + barrier per chunk).
//   - GELU on H2 applied via thread_elements() in registers (no SMEM round-trip
//     before layer 3).
//   - Layer 3 kept as scalar (faster than MMA staging overhead at this size).
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Exact-erf GELU using Abramowitz 7.1.26 with fast::exp. Within 1e-2 of MLX nn.gelu.
static inline float gelu_exact(float v) {
    const float x = v * 0.70710678118f;  // x/sqrt(2)
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * fast::exp(-ax * ax);
    return 0.5f * v * (1.0f + sign * y);
}

constant constexpr uint M_    = 16;
constant constexpr uint D1    = 128;
constant constexpr uint D2    = 512;
constant constexpr uint DO    = 10;
constant constexpr uint CHUNK = 128;
constant constexpr uint N_CHUNKS = D2 / CHUNK;  // 4

kernel void mlp_f32(
    device const float* x   [[buffer(0)]],
    device const float* W1  [[buffer(1)]],
    device const float* W2  [[buffer(2)]],
    device const float* W3  [[buffer(3)]],
    device       float* y   [[buffer(4)]],
    constant     uint& N    [[buffer(5)]],
    constant     uint& _D1  [[buffer(6)]],
    constant     uint& _D2  [[buffer(7)]],
    constant     uint& _DO  [[buffer(8)]],
    uint  tid               [[thread_position_in_threadgroup]],
    uint  sgid              [[simdgroup_index_in_threadgroup]])
{
    threadgroup float xs[M_ * D1];
    threadgroup float h1c[M_ * CHUNK];
    threadgroup float h2[M_ * D1];

    const uint row_tile = sgid >> 4;
    const uint col_tile = sgid & 15u;
    const uint row0 = row_tile * 8;
    const uint col0 = col_tile * 8;

    xs[tid]        = x[tid];
    xs[tid + 1024] = x[tid + 1024];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> H2(0.0f);

    for (uint ch = 0; ch < N_CHUNKS; ++ch) {
        const uint chunk_col_base = ch * CHUNK;

        // ---- Layer 1 chunk: C = xs @ W1[:, chunk_col_base + col0], then GELU in regs.
        {
            simdgroup_matrix<float, 8, 8> C(0.0f);
            for (uint k0 = 0; k0 < D1; k0 += 8) {
                simdgroup_matrix<float, 8, 8> A, B;
                simdgroup_load(A, xs + row0 * D1 + k0,                              D1, ulong2(0, 0));
                simdgroup_load(B, W1 + k0 * D2 + chunk_col_base + col0,             D2, ulong2(0, 0));
                simdgroup_multiply_accumulate(C, A, B, C);
            }
            thread auto& ce = C.thread_elements();
            constexpr uint NE = sizeof(ce) / sizeof(float);
            thread float* cf = (thread float*)&ce;
            for (uint i = 0; i < NE; ++i) cf[i] = gelu_exact(cf[i]);
            simdgroup_store(C, h1c + row0 * CHUNK + col0, CHUNK, ulong2(0, 0));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- H2 += h1c @ W2[chunk_rows, :]
        {
            for (uint k0 = 0; k0 < CHUNK; k0 += 8) {
                simdgroup_matrix<float, 8, 8> A, B;
                simdgroup_load(A, h1c + row0 * CHUNK + k0,                              CHUNK, ulong2(0, 0));
                simdgroup_load(B, W2 + (chunk_col_base + k0) * D1 + col0,               D1,    ulong2(0, 0));
                simdgroup_multiply_accumulate(H2, A, B, H2);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- GELU on H2 in registers, then store to h2 SMEM.
    {
        thread auto& he = H2.thread_elements();
        constexpr uint NE = sizeof(he) / sizeof(float);
        thread float* hf = (thread float*)&he;
        for (uint i = 0; i < NE; ++i) hf[i] = gelu_exact(hf[i]);
    }
    simdgroup_store(H2, h2 + row0 * D1 + col0, D1, ulong2(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Layer 3 scalar: (16 x 128) @ (128 x 10) = (16 x 10).
    if (tid < M_ * DO) {
        uint r = tid / DO;
        uint c = tid % DO;
        float s = 0.0f;
        for (uint k = 0; k < D1; k += 4) {
            s += h2[r * D1 + k    ] * W3[(k    ) * DO + c];
            s += h2[r * D1 + k + 1] * W3[(k + 1) * DO + c];
            s += h2[r * D1 + k + 2] * W3[(k + 2) * DO + c];
            s += h2[r * D1 + k + 3] * W3[(k + 3) * DO + c];
        }
        y[r * DO + c] = s;
    }
}
