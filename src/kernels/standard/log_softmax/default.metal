// log_softmax: y = x - logsumexp(x). One TG per row. C == TG == 1024.
#include <metal_stdlib>
using namespace metal;

kernel void log_softmax_f32(
    device const float*  X       [[buffer(0)]],
    device       float*  Y       [[buffer(1)]],
    constant     uint&   C       [[buffer(2)]],
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint tid = tid3.x;
    const uint row = tgid.y;
    const uint lane = tid & 31u;
    const uint sg = tid >> 5;
    device const float* xr = X + row * C;
    device       float* yr = Y + row * C;

    threadgroup float reduce[32];

    float v = xr[tid];

    // max
    float mx = simd_max(v);
    if (lane == 0) reduce[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float m = simd_max(reduce[lane]);
        if (lane == 0) reduce[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = reduce[0];

    // exp + sum
    float e = fast::exp(v - row_max);
    float sum = simd_sum(e);
    if (lane == 0) reduce[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float s = simd_sum(reduce[lane]);
        if (lane == 0) reduce[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lse = row_max + log(reduce[0]);

    yr[tid] = v - lse;
}
