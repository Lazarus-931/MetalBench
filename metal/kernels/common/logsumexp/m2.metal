// logsumexp M2 variant: per-row log(sum(exp(x - max))) + max.
// Strategy: single simdgroup (32 threads) per row, C=1024 per row.
// Each lane owns 32 contiguous elements loaded via float4 (8 vec loads).
// Online single-pass logsumexp combines max + sum: maintain (m, s) such that
// s = sum(exp(x_i - m)). When merging, the larger m wins and the other side
// is rescaled by exp(delta). Final result = m + log(s). One pass over memory.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint TG_SIZE      = 32u;
constant constexpr uint C_PER_LANE   = 32u;   // 1024 / 32
constant constexpr uint VEC_PER_LANE = 8u;    // 32 / 4

kernel void logsumexp_f32(
    device const float*  x       [[buffer(0)]],
    device       float*  y       [[buffer(1)]],
    constant     uint&   C       [[buffer(2)]],
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]],
    uint  simd_lane             [[thread_index_in_simdgroup]])
{
    (void)C;
    const uint tid = tid3.x;
    if (tid >= TG_SIZE) return;
    const uint row = tgid.y;

    device const float4* xr4 = (device const float4*)(x + row * 1024u);
    const uint base_vec = simd_lane * VEC_PER_LANE;

    // Local online (m, s): start with m = first elem, s = 1.
    float4 q0 = xr4[base_vec + 0u];
    float lm = fmax(fmax(q0.x, q0.y), fmax(q0.z, q0.w));
    float ls = (fast::exp(q0.x - lm) + fast::exp(q0.y - lm))
             + (fast::exp(q0.z - lm) + fast::exp(q0.w - lm));

    #pragma unroll
    for (uint j = 1u; j < VEC_PER_LANE; ++j) {
        float4 q = xr4[base_vec + j];
        float nm = fmax(fmax(q.x, q.y), fmax(q.z, q.w));
        float ns = (fast::exp(q.x - nm) + fast::exp(q.y - nm))
                 + (fast::exp(q.z - nm) + fast::exp(q.w - nm));
        // Merge (lm, ls) and (nm, ns).
        float m = fmax(lm, nm);
        ls = ls * fast::exp(lm - m) + ns * fast::exp(nm - m);
        lm = m;
    }

    // Cross-lane reduction via simdgroup shuffles, online merge.
    #pragma unroll
    for (uint off = 16u; off > 0u; off >>= 1) {
        float om = simd_shuffle_xor(lm, off);
        float os = simd_shuffle_xor(ls, off);
        float m  = fmax(lm, om);
        ls = ls * fast::exp(lm - m) + os * fast::exp(om - m);
        lm = m;
    }

    if (simd_lane == 0u) {
        y[row] = lm + fast::log(ls);
    }
}
