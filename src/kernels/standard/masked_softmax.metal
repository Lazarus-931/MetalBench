// masked_softmax: y = softmax(x + mask). Per-row.
#include <metal_stdlib>
using namespace metal;

kernel void masked_softmax_f32(
    device const float*  X       [[buffer(0)]],
    device const float*  M       [[buffer(1)]],
    device       float*  Y       [[buffer(2)]],
    constant     uint&   C       [[buffer(3)]],
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint TG = 1024;
    const uint tid = tid3.x;
    const uint row = tgid.y;
    device const float* xr = X + row * C;
    device const float* mr = M + row * C;
    device       float* yr = Y + row * C;

    threadgroup float reduce[32];

    float mx = -INFINITY;
    for (uint i = tid; i < C; i += TG) mx = fmax(mx, xr[i] + mr[i]);
    mx = simd_max(mx);
    if ((tid & 31) == 0) reduce[tid >> 5] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        mx = simd_max(reduce[tid]);
        if (tid == 0) reduce[0] = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = reduce[0];

    float sum = 0.0f;
    for (uint i = tid; i < C; i += TG) {
        float e = fast::exp(xr[i] + mr[i] - row_max);
        yr[i] = e;  // store for second pass to avoid recompute
        sum += e;
    }
    sum = simd_sum(sum);
    if ((tid & 31) == 0) reduce[tid >> 5] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        sum = simd_sum(reduce[tid]);
        if (tid == 0) reduce[0] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.0f / reduce[0];

    for (uint i = tid; i < C; i += TG) yr[i] = yr[i] * inv;
}
