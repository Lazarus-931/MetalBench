// top_k M2 variant: per-row top-k sorted desc. R=1024, C=1024, k=16.
// Strategy: TG=32 threads (single simdgroup). Each thread owns C/32 = 32
// elements loaded via float4. Maintain per-lane running max + idx in
// registers. Each of k iterations does a single simd_max + ballot to pick
// the row-global winner, then the winning lane invalidates its slot and
// rescans its 32-element register file for a new local max. Avoids any
// threadgroup memory traffic and any barriers.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint C_PER_LANE = 32;        // 1024 / 32
constant constexpr uint VEC_PER_LANE = 8;       // 32 / 4

kernel void top_k_f32(
    device const float*  x       [[buffer(0)]],
    device       float*  y       [[buffer(1)]],
    constant     uint&   C       [[buffer(2)]],
    constant     uint&   k       [[buffer(3)]],
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]],
    uint  simd_lane             [[thread_index_in_simdgroup]],
    uint  simd_id               [[simdgroup_index_in_threadgroup]])
{
    (void)C; (void)k; (void)simd_id;
    const uint row = tgid.y;
    const uint tid = tid3.x;
    if (tid >= 32u) return;

    // Each lane owns 32 contiguous elements: x[row, lane*32 .. lane*32+31].
    device const float4* xr4 = (device const float4*)(x + row * 1024u);
    const uint base_vec = simd_lane * VEC_PER_LANE; // 0..256

    float v[C_PER_LANE];
    #pragma unroll
    for (uint j = 0; j < VEC_PER_LANE; ++j) {
        float4 q = xr4[base_vec + j];
        v[j*4 + 0] = q.x;
        v[j*4 + 1] = q.y;
        v[j*4 + 2] = q.z;
        v[j*4 + 3] = q.w;
    }

    // Initial scan: find local max value + index.
    float lmax = v[0];
    uint  lidx = 0u;
    #pragma unroll
    for (uint j = 1; j < C_PER_LANE; ++j) {
        if (v[j] > lmax) { lmax = v[j]; lidx = j; }
    }

    device float* yr = y + row * 16u;

    // Pull top-16 across the simdgroup.
    #pragma unroll
    for (uint out_i = 0; out_i < 16u; ++out_i) {
        float gmx = simd_max(lmax);
        bool is_winner = (lmax == gmx);
        simd_vote vote = simd_ballot(is_winner);
        ulong mask = (ulong)vote;
        uint  win_lane = (uint)ctz(mask);

        if (simd_lane == 0u) {
            yr[out_i] = gmx;
        }

        if (simd_lane == win_lane) {
            // Invalidate this slot and rescan locally for a new max.
            v[lidx] = -INFINITY;
            float nm = v[0];
            uint  ni = 0u;
            #pragma unroll
            for (uint j = 1; j < C_PER_LANE; ++j) {
                if (v[j] > nm) { nm = v[j]; ni = j; }
            }
            lmax = nm;
            lidx = ni;
        }
    }
}
