// top_k: per-row top-k values, sorted descending. R=1024, C=1024, k=16.
// Strategy: 32 threads per row. Each thread holds 32 elements in registers
// (loaded via float4). Threads write their best (current local max) to
// threadgroup memory each iteration; thread 0 reduces the 32 candidates to
// pick the row-global winner, broadcasts the winning lane id, then the
// winning lane invalidates its slot and rescans its registers.
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
    (void)C; (void)k; (void)simd_id; (void)simd_lane;
    const uint row = tgid.y;
    const uint tid = tid3.x;

    threadgroup float t_vals[32];
    threadgroup uint  t_win;

    if (tid >= 32u) return;

    device const float4* xr4 = (device const float4*)(x + row * 1024u);
    const uint base_vec = tid * VEC_PER_LANE;

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

    for (uint out_i = 0; out_i < 16u; ++out_i) {
        t_vals[tid] = lmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float bv = t_vals[0];
            uint  bi = 0u;
            for (uint j = 1; j < 32u; ++j) {
                float vj = t_vals[j];
                if (vj > bv) { bv = vj; bi = j; }
            }
            yr[out_i] = bv;
            t_win = bi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint win = t_win;
        if (tid == win) {
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
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
