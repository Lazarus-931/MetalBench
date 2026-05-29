// nll_loss: per-row -sum(y_onehot * log_probs). Output (N,).
// Optimized: float4 vector loads, 256 active threads handle 4 elements each,
// single-simdgroup-per-row final reduction across 8 subgroup partials.
#include <metal_stdlib>
using namespace metal;

kernel void nll_loss_f32(
    device const float*  LP       [[buffer(0)]],
    device const float*  Yh       [[buffer(1)]],
    device       float*  out      [[buffer(2)]],
    constant     uint&   N        [[buffer(3)]],
    constant     uint&   C        [[buffer(4)]],
    uint3 tid3                   [[thread_position_in_threadgroup]],
    uint3 tgid                   [[threadgroup_position_in_grid]])
{
    const uint tid  = tid3.x;
    const uint row  = tgid.y;
    const uint lane = tid & 31u;
    const uint sgid = tid >> 5;

    threadgroup float reduce[8];

    float partial = 0.0f;
    if (tid < 256u) {
        const uint base = row * C + (tid << 2);
        device const float4* lpp = reinterpret_cast<device const float4*>(LP + base);
        device const float4* yhp = reinterpret_cast<device const float4*>(Yh + base);
        float4 lv = *lpp;
        float4 yv = *yhp;
        float4 prod = lv * yv;
        partial = -(prod.x + prod.y + prod.z + prod.w);
    }

    float sg_sum = simd_sum(partial);
    if (tid < 256u && lane == 0u) reduce[sgid] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float v = (lane < 8u) ? reduce[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) out[row] = v;
    }
}
