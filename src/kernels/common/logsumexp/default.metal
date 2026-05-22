// logsumexp: per-row log(sum(exp(x - max))) + max. One TG per row, 1024 threads.
#include <metal_stdlib>
using namespace metal;

kernel void logsumexp_f32(
    device const float*  x       [[buffer(0)]],
    device       float*  y       [[buffer(1)]],
    constant     uint&   C       [[buffer(2)]],
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint TG = 1024;
    const uint tid = tid3.x;
    const uint row = tgid.y;
    device const float* xr = x + row * C;

    threadgroup float reduce[32];

    float mx = -INFINITY;
    for (uint i = tid; i < C; i += TG) mx = fmax(mx, xr[i]);
    mx = simd_max(mx);
    if ((tid & 31) == 0) reduce[tid >> 5] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        mx = reduce[tid];
        mx = simd_max(mx);
        if (tid == 0) reduce[0] = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = reduce[0];

    float s = 0.0f;
    for (uint i = tid; i < C; i += TG) s += fast::exp(xr[i] - row_max);
    s = simd_sum(s);
    if ((tid & 31) == 0) reduce[tid >> 5] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        s = reduce[tid];
        s = simd_sum(s);
        if (tid == 0) y[row] = row_max + fast::log(reduce[0]);
    }
}
