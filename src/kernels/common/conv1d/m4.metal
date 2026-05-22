// conv1d: NLC tiled with threadgroup-memory input cache.
// x (N=8, L=256, C=64), w (K=128, R=3, C=64), y (N, L2=254, K).
//
// Tile: each TG processes one (n, L2_block of 32 positions) × all K=128 = 4096 outputs.
// Thread mapping: 1024 threads × 4 outputs each = 4096.
// Per-thread layout: thread t handles outputs (l_in_tile, k) for 4 (l_in_tile,k) pairs.
//
// We use 32 l × 128 k with t = l*32 + k_quarter? Simpler: each thread takes a contiguous
// quad along K. Thread t: l = t / 32, k_base = (t % 32) * 4 → k in {k_base..k_base+3}.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint TILE_L  = 32;
constant constexpr uint OUT_K   = 128;
constant constexpr uint K_INNER = 64;     // C
constant constexpr uint R_K     = 3;
constant constexpr uint STRIP_L = TILE_L + R_K - 1;  // 34
constant constexpr uint STRIP_ELEMS = STRIP_L * K_INNER;  // 2176

kernel void conv1d_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  w       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   N       [[buffer(3)]],
    constant     uint&   C       [[buffer(4)]],
    constant     uint&   L       [[buffer(5)]],
    constant     uint&   K       [[buffer(6)]],
    constant     uint&   R       [[buffer(7)]],
    constant     uint&   stride  [[buffer(8)]],
    uint tid_in_tg [[thread_position_in_threadgroup]],
    uint tg_id     [[threadgroup_position_in_grid]])
{
    const uint L2 = (L - R) / stride + 1;            // 254
    const uint NUM_L_TILES = (L2 + TILE_L - 1) / TILE_L;  // 8
    const uint TILES = N * NUM_L_TILES;               // 64
    const uint NUM_TGS = 64;

    threadgroup float xstrip[STRIP_ELEMS];

    // Thread decomposition: 1024 = 32 l × 32 k4-groups.
    // We hold 4 outputs per thread: along K dim (k_base..k_base+3).
    const uint t_l = tid_in_tg / 32u;        // 0..31  (l in tile)
    const uint t_k4 = tid_in_tg % 32u;       // 0..31  (k4 group)

    for (uint tile = tg_id; tile < TILES; tile += NUM_TGS) {
        uint n = tile / NUM_L_TILES;
        uint lt = tile % NUM_L_TILES;
        uint l2_base = lt * TILE_L;
        uint l2_count = min(TILE_L, L2 - l2_base);
        // Strip window length
        uint l_window = min(STRIP_L, L - l2_base);

        // Load x[n, l2_base : l2_base+STRIP_L, 0:C] into xstrip.
        // STRIP_ELEMS=2176; 1024 threads, vec4 → 544 vec4. 1 vec4 per thread, partial.
        device const float4* xv4 = (device const float4*)(x + (n * L + l2_base) * C);
        threadgroup float4* sv4 = (threadgroup float4*)xstrip;
        const uint VEC_COUNT = STRIP_ELEMS / 4u;     // 544
        const uint VALID_VEC = (l_window * K_INNER) / 4u;
        for (uint i = tid_in_tg; i < VEC_COUNT; i += 1024u) {
            sv4[i] = (i < VALID_VEC) ? xv4[i] : float4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (t_l < l2_count) {
            // Compute 4 outputs (l = l2_base + t_l, k in [t_k4*4 .. t_k4*4+3]).
            uint k_base = t_k4 * 4u;
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

            // Iterate rr ∈ {0,1,2}, cc over C (16 vec4).
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < R_K; ++rr) {
                uint xs_base = (t_l + rr) * K_INNER;
                threadgroup const float4* xv = (threadgroup const float4*)(xstrip + xs_base);
                // For each of 4 k channels, fetch its weight row w[k, rr, :].
                // w stride: w[k, rr, c] = w + (k * R + rr) * C + c
                device const float4* w0 = (device const float4*)(w + ((k_base + 0u) * R_K + rr) * K_INNER);
                device const float4* w1 = (device const float4*)(w + ((k_base + 1u) * R_K + rr) * K_INNER);
                device const float4* w2v = (device const float4*)(w + ((k_base + 2u) * R_K + rr) * K_INNER);
                device const float4* w3 = (device const float4*)(w + ((k_base + 3u) * R_K + rr) * K_INNER);
                #pragma clang loop unroll(full)
                for (uint cc = 0; cc < 16u; ++cc) {
                    float4 a = xv[cc];
                    float4 b0 = w0[cc];
                    float4 b1 = w1[cc];
                    float4 b2 = w2v[cc];
                    float4 b3 = w3[cc];
                    sum0 += a.x*b0.x + a.y*b0.y + a.z*b0.z + a.w*b0.w;
                    sum1 += a.x*b1.x + a.y*b1.y + a.z*b1.z + a.w*b1.w;
                    sum2 += a.x*b2.x + a.y*b2.y + a.z*b2.z + a.w*b2.w;
                    sum3 += a.x*b3.x + a.y*b3.y + a.z*b3.z + a.w*b3.w;
                }
            }
            uint l2 = l2_base + t_l;
            device float* yptr = y + (n * L2 + l2) * K + k_base;
            yptr[0] = sum0;
            yptr[1] = sum1;
            yptr[2] = sum2;
            yptr[3] = sum3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
