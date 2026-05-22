// nll_loss: per-row -sum(y_onehot * log_probs). Output (N,).
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
    const uint tid = tid3.x;
    const uint row = tgid.y;
    const uint lane = tid & 31u;
    const uint sgid = tid >> 5;
    const uint base = row * C + tid;

    threadgroup float reduce[32];

    float nll = -LP[base] * Yh[base];
    nll = simd_sum(nll);
    if (lane == 0) reduce[sgid] = nll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = simd_sum(reduce[lane]);
        if (lane == 0) out[row] = v;
    }
}
