#include <metal_stdlib>
using namespace metal;

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
    device const float* ar = a + row * D;
    device const float* pr = p + row * D;
    device const float* nr = n + row * D;
    threadgroup float r1[32], r2[32];

    float av = ar[tid];
    float dp = av - pr[tid];
    float dn = av - nr[tid];
    float sp = dp * dp;
    float sn = dn * dn;

    sp = simd_sum(sp);
    sn = simd_sum(sn);
    if (lane == 0) { r1[sgid] = sp; r2[sgid] = sn; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float vp = simd_sum(r1[lane]);
        float vn = simd_sum(r2[lane]);
        if (lane == 0) y[row] = fmax(sqrt(vp) - sqrt(vn) + margin, 0.0f);
    }
}
