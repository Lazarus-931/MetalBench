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

    threadgroup float reduce_mx[32];
    threadgroup float reduce_s[32];
    threadgroup float row_max_tg;
    threadgroup float row_sum_tg;

    float mx = -INFINITY;
    for (uint i = tid; i < C; i += TG) mx = fmax(mx, xr[i]);
    mx = simd_max(mx);
    if ((tid & 31) == 0) reduce_mx[tid >> 5] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = reduce_mx[tid];
        v = simd_max(v);
        if (tid == 0) row_max_tg = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = row_max_tg;

    float s = 0.0f;
    for (uint i = tid; i < C; i += TG) s += precise::exp(xr[i] - row_max);
    s = simd_sum(s);
    if ((tid & 31) == 0) reduce_s[tid >> 5] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = reduce_s[tid];
        v = simd_sum(v);
        if (tid == 0) row_sum_tg = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) y[row] = row_max + precise::log(row_sum_tg);
}
