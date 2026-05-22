// log_softmax_cross_entropy: fused log_softmax + NLL. Output (N,).
#include <metal_stdlib>
using namespace metal;

kernel void log_softmax_cross_entropy_f32(
    device const float*  L        [[buffer(0)]],
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
    device const float* lp = L  + row * C;
    device const float* yp = Yh + row * C;

    threadgroup float reduce[32];
    threadgroup float scalars[2];

    float lv = lp[tid];
    float yv = yp[tid];

    float mx = simd_max(lv);
    if (lane == 0) reduce[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = simd_max(reduce[lane]);
        if (lane == 0) scalars[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = scalars[0];

    float e = precise::exp(lv - row_max);
    float s = simd_sum(e);
    if (lane == 0) reduce[sgid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = simd_sum(reduce[lane]);
        if (lane == 0) scalars[1] = row_max + precise::log(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lse = scalars[1];

    float nll = yv * (lse - lv);
    float ns = simd_sum(nll);
    if (lane == 0) reduce[sgid] = ns;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = simd_sum(reduce[lane]);
        if (lane == 0) out[row] = v;
    }
}
