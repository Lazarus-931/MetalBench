// top_k: per-row top-k values, sorted descending. R=1024, C=1024, k=16.
#include <metal_stdlib>
using namespace metal;

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
    const uint tid = tid3.x;
    const uint row = tgid.y;
    device const float* xr = x + row * C;
    device       float* yr = y + row * k;

    // Each thread owns one column value.
    float v = (tid < C) ? xr[tid] : -INFINITY;

    // Per-simd partial maxes and the winning simd id for each iteration.
    // Layout: smax[0..31] = per-simd max; smax[32] = global max (broadcast);
    // win_simd[0] = id of the simd that owns the global max (lowest-id wins ties).
    threadgroup float smax[33];
    threadgroup uint  win_simd[1];

    // 16 iterations. Each iter:
    //  A) every simd reduces its 32 lanes -> 32 partials.
    //  B) simd 0 reduces 32 partials -> global max + winning simd id.
    //  C) only the winning simd masks one of its lanes that equals gmx.
    for (uint out_i = 0; out_i < 16; ++out_i) {
        // Stage A: per-simd max.
        float sm = simd_max(v);
        if (simd_lane == 0) {
            smax[simd_id] = sm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stage B: simd 0 reduces 32 partials.
        if (simd_id == 0) {
            float p = smax[simd_lane];        // 32 lanes load 32 partials.
            float g = simd_max(p);
            // Find lowest-lane index with p==g.
            simd_vote vv = simd_ballot(p == g);
            uint w = (uint)ctz((ulong)vv);
            if (simd_lane == 0) {
                smax[32]    = g;
                win_simd[0] = w;
                yr[out_i]   = g;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float gmx     = smax[32];
        uint  win_sid = win_simd[0];

        // Stage C: only the winning simd masks exactly one lane equal to gmx.
        if (simd_id == win_sid) {
            simd_vote vote = simd_ballot(v == gmx);
            uint wl = (uint)ctz((ulong)vote);
            if (simd_lane == wl) {
                v = -INFINITY;
            }
        }
    }
}
