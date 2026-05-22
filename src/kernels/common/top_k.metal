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

    float v = (tid < C) ? xr[tid] : -INFINITY;

    threadgroup float cand[512];

    float cur = v;
    for (uint i = 0; i < 16; ++i) {
        float mx = simd_max(cur);
        bool is_winner = (cur == mx);
        simd_vote vote = simd_ballot(is_winner);
        ulong mask = (ulong)vote;
        uint win_lane = (uint)ctz(mask);
        if (simd_lane == win_lane) {
            cand[simd_id * 16 + i] = cur;
            cur = -INFINITY;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float local[16];
        for (uint i = 0; i < 16; ++i) {
            local[i] = cand[i * 32 + simd_lane];
        }
        for (uint out_i = 0; out_i < 16; ++out_i) {
            float lmax = local[0];
            uint lmax_idx = 0;
            for (uint i = 1; i < 16; ++i) {
                if (local[i] > lmax) { lmax = local[i]; lmax_idx = i; }
            }
            float gmx = simd_max(lmax);
            bool is_winner = (lmax == gmx);
            simd_vote vote = simd_ballot(is_winner);
            ulong mask = (ulong)vote;
            uint win_lane = (uint)ctz(mask);
            if (simd_lane == win_lane) {
                local[lmax_idx] = -INFINITY;
            }
            if (simd_lane == 0) {
                yr[out_i] = gmx;
            }
        }
    }
}
