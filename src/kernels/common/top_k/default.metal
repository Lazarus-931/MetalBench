// top_k: per-row top-k values, sorted descending. R=1024, C=1024, k=16.
// Strategy: single simdgroup (32 lanes) per row. Each lane owns 32 elements
// (C_PER_LANE = 1024/32). Each iteration: simd_max picks row-global max; we
// resolve ties by lane id via a tagged simd_max on (uint)(~lane). The winning
// lane invalidates its slot and rescans its 32 registers. No threadgroup
// memory, no barriers.
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

    device const float4* xr4 = (device const float4*)(x + row * 1024u);
    const uint base_vec = simd_lane * VEC_PER_LANE;

    float v[C_PER_LANE];
    #pragma unroll
    for (uint j = 0; j < VEC_PER_LANE; ++j) {
        float4 q = xr4[base_vec + j];
        v[j*4 + 0] = q.x;
        v[j*4 + 1] = q.y;
        v[j*4 + 2] = q.z;
        v[j*4 + 3] = q.w;
    }

    float lmax = v[0];
    uint  lidx = 0u;
    #pragma unroll
    for (uint j = 1; j < C_PER_LANE; ++j) {
        if (v[j] > lmax) { lmax = v[j]; lidx = j; }
    }

    device float* yr = y + row * 16u;

    #pragma unroll
    for (uint out_i = 0; out_i < 16u; ++out_i) {
        float gmx = simd_max(lmax);
        // Tie-break: among lanes where lmax==gmx, pick the lowest lane id.
        // Encode (lmax==gmx ? lane : 32) and take simd_min.
        uint tag = (lmax == gmx) ? simd_lane : 32u;
        uint win_lane = simd_min(tag);

        if (simd_lane == 0u) {
            yr[out_i] = gmx;
        }

        if (simd_lane == win_lane) {
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
