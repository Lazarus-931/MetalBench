// mlp on M4: simdgroup_matrix MMA, fused layer1->GELU->layer2 across D2 chunks.
// Shapes: x(16,128) @ W1(128,512) -> GELU -> @ W2(512,128) -> GELU -> @ W3(128,10) -> y(16,10)
//
// Strategy:
//   - Single pass over the M=16 batch (no row tiling).
//   - Don't materialize the full 16x512 h1; instead chunk D2 into CHUNK=128 columns.
//   - For each chunk: compute h1_chunk = GELU(xs @ W1[:, chunk]), then accumulate
//     h2 += h1_chunk @ W2[chunk_rows, :].
//   - 1 threadgroup, 1024 threads = 32 simdgroups, each owns one 8x8 output tile of
//     a 16x128 grid (2 row tiles x 16 col tiles = 32 tiles).
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

constant constexpr uint M_   = 16;
constant constexpr uint D1   = 128;
constant constexpr uint D2   = 512;
constant constexpr uint DO   = 10;
constant constexpr uint CHUNK = 128;       // columns of D2 processed per iteration
constant constexpr uint N_CHUNKS = D2 / CHUNK;  // 4

// TG memory:
//   xs:       M*D1     = 16*128 = 2048 floats =  8KB
//   h1_chunk: M*CHUNK  = 16*128 = 2048 floats =  8KB
//   h2:       M*D1     = 16*128 = 2048 floats =  8KB
// Total: 24KB.

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

    // Each simdgroup owns one 8x8 output tile of a 16x128 grid:
    //   row_tile in {0,1} (sgid >> 4), col_tile in {0..15} (sgid & 15).
    const uint row_tile = sgid >> 4;       // 0 or 1
    const uint col_tile = sgid & 15u;      // 0..15
    const uint row0 = row_tile * 8;
    const uint col0 = col_tile * 8;

    // ---- Load xs (16*128 = 2048 floats); 1024 threads -> 2 per thread.
    xs[tid]        = x[tid];
    xs[tid + 1024] = x[tid + 1024];
    // ---- Zero h2.
    h2[tid]        = 0.0f;
    h2[tid + 1024] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // h2 accumulator for this SG's tile, kept in registers across chunks.
    simdgroup_matrix<float, 8, 8> H2(0.0f);

    for (uint ch = 0; ch < N_CHUNKS; ++ch) {
        const uint chunk_col_base = ch * CHUNK;  // W1 column offset, W2 row offset

        // ---- Layer 1 (chunked along D2): h1c_tile = xs @ W1[:, chunk_col_base + col0]
        //   xs is (16 x D1), W1 slice is (D1 x CHUNK). Each SG produces an 8x8 tile.
        {
            simdgroup_matrix<float, 8, 8> C(0.0f);
            for (uint k0 = 0; k0 < D1; k0 += 8) {
                simdgroup_matrix<float, 8, 8> A, B;
                simdgroup_load(A, xs + row0 * D1 + k0,                              D1, ulong2(0, 0));
                simdgroup_load(B, W1 + k0 * D2 + chunk_col_base + col0,             D2, ulong2(0, 0));
                simdgroup_multiply_accumulate(C, A, B, C);
            }
            // Store directly into h1c (16 x CHUNK).
            simdgroup_store(C, h1c + row0 * CHUNK + col0, CHUNK, ulong2(0, 0));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- GELU on h1c (16*CHUNK = 2048 floats); 2 per thread.
        h1c[tid]        = gelu_exact(h1c[tid]);
        h1c[tid + 1024] = gelu_exact(h1c[tid + 1024]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Accumulate into H2: H2 += h1c @ W2[chunk_rows, :]
        // h1c is (16 x CHUNK), W2 slice is (CHUNK x D1). Each SG owns an 8x8 of (16 x D1).
        {
            for (uint k0 = 0; k0 < CHUNK; k0 += 8) {
                simdgroup_matrix<float, 8, 8> A, B;
                simdgroup_load(A, h1c + row0 * CHUNK + k0,                              CHUNK, ulong2(0, 0));
                simdgroup_load(B, W2 + (chunk_col_base + k0) * D1 + col0,               D1,    ulong2(0, 0));
                simdgroup_multiply_accumulate(H2, A, B, H2);
            }
        }
        // Only need a barrier before next chunk overwrites h1c. Skip on last iter.
        if (ch + 1 < N_CHUNKS) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // ---- Store H2 to threadgroup memory and apply GELU.
    simdgroup_store(H2, h2 + row0 * D1 + col0, D1, ulong2(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // GELU on h2 (16*128 = 2048; 2 per thread)
    h2[tid]        = gelu_exact(h2[tid]);
    h2[tid + 1024] = gelu_exact(h2[tid + 1024]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Layer 3: (16 x 128) @ (128 x 10) = (16 x 10). Parallel reduce: 1024 threads
    //   = 160 outputs * 6.4 partials. Use simdgroup_sum over 16-thread K-shards.
    //   Partition each (r,c) across 8 lanes -> 1280 lanes used. tid -> (r,c,k_shard).
    // Simpler: each of 160 outputs gets 4 threads accumulating D1/4=32 elements, sum.
    // Layer 3: 160 outputs * 4 K-shards = 640 lanes. Each shard does 32 elems.
    {
        if (tid < M_ * DO * 4) {
            uint k_shard = tid & 3u;
            uint idx = tid >> 2;
            uint r = idx / DO;
            uint c = idx % DO;
            uint k0 = k_shard * 32;
            float s = 0.0f;
            for (uint k = k0; k < k0 + 32; k += 4) {
                s += h2[r * D1 + k    ] * W3[(k    ) * DO + c];
                s += h2[r * D1 + k + 1] * W3[(k + 1) * DO + c];
                s += h2[r * D1 + k + 2] * W3[(k + 2) * DO + c];
                s += h2[r * D1 + k + 3] * W3[(k + 3) * DO + c];
            }
            threadgroup float* scratch = h1c;
            scratch[idx * 4 + k_shard] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < M_ * DO) {
            threadgroup float* scratch = h1c;
            float s = scratch[tid * 4] + scratch[tid * 4 + 1] + scratch[tid * 4 + 2] + scratch[tid * 4 + 3];
            uint r = tid / DO;
            uint c = tid % DO;
            y[r * DO + c] = s;
        }
    }
}
