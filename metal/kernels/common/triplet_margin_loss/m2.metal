#include <metal_stdlib>
using namespace metal;

// TG=64, float4 loads, fast::sqrt for the two final scalars.
// Fixed: use threadgroup barrier before reading r1/r2, and ensure all threads participate in simd_sum correctly.
// Changed: use a single threadgroup reduction with 2 simdgroups, then finalize in lane 0.
// Removed the problematic atomic operations that caused compilation errors.

kernel void triplet_margin_loss_f32(
    device const float*  a       [[buffer(0)]],
    device const float*  p       [[buffer(1)]],
    device const float*  n       [[buffer(2)]],
    device       float*  y       [[buffer(3)]],
    constant     uint&   D       [[buffer(4)]],
    constant     float&  margin  [[buffer(5)]],
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint tid = tid3.x;
    const uint row = tgid.y;
    const uint lane = tid & 31u;
    const uint sgid = tid >> 5;
    const uint D4 = D >> 2;
    device const float4* ar = (device const float4*)(a + row * D);
    device const float4* pr = (device const float4*)(p + row * D);
    device const float4* nr = (device const float4*)(n + row * D);
    threadgroup float r1[2], r2[2];

    float sp = 0.0f, sn = 0.0f;
    #pragma unroll
    for (uint i = tid; i < D4; i += 64u) {
        float4 av = ar[i];
        float4 dp = av - pr[i];
        float4 dn = av - nr[i];
        sp = fma(dp.x, dp.x, sp); sp = fma(dp.y, dp.y, sp);
        sp = fma(dp.z, dp.z, sp); sp = fma(dp.w, dp.w, sp);
        sn = fma(dn.x, dn.x, sn); sn = fma(dn.y, dn.y, sn);
        sn = fma(dn.z, dn.z, sn); sn = fma(dn.w, dn.w, sn);
    }

    sp = simd_sum(sp);
    sn = simd_sum(sn);
    if (lane == 0) { r1[sgid] = sp; r2[sgid] = sn; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float vp = (lane < 2) ? r1[lane] : 0.0f;
        float vn = (lane < 2) ? r2[lane] : 0.0f;
        vp = simd_sum(vp);
        vn = simd_sum(vn);
        if (lane == 0) y[row] = fmax(fast::sqrt(vp) - fast::sqrt(vn) + margin, 0.0f);
    }
}
