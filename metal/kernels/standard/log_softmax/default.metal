// log_softmax: y = x - logsumexp(x). One TG per row. C == TG == 1024.
// 32 simdgroups; two-stage reduce via threadgroup scratch.
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

    threadgroup float scratch_max[32];
    threadgroup float scratch_sum[32];

    float v = xr[tid];

    // Row max
    float mx = simd_max(v);
    if (lane == 0) scratch_max[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = simd_max(scratch_max[lane]);

    // Row sum of exp (separate scratch -> no barrier needed between writes)
    float e = fast::exp(v - row_max);
    float sum = simd_sum(e);
    if (lane == 0) scratch_sum[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_sum = simd_sum(scratch_sum[lane]);

    float lse = row_max + fast::log(row_sum);
    yr[tid] = v - lse;
}
