// softmax per row: simd_max + simd_sum reductions.
#include <metal_stdlib>
using namespace metal;

kernel void softmax_f32(
    device const float*  x  [[buffer(0)]],
    device       float*  y  [[buffer(1)]],
    constant     uint&   N  [[buffer(2)]],
    uint3 tid              [[thread_position_in_threadgroup]],
    uint3 tgid             [[threadgroup_position_in_grid]])
{
    const uint t = tid.x;
    const uint row = tgid.y;
    const uint off = row * N;
    const uint sg = t >> 5;
    const uint lane = t & 31;

    threadgroup float tg_buf[32];
    threadgroup float row_max;
    threadgroup float row_sum;

    float val = x[off + t];

    float m = simd_max(val);
    if (lane == 0) tg_buf[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = tg_buf[lane];
        v = simd_max(v);
        if (lane == 0) row_max = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rmax = row_max;

    float ev = fast::exp(val - rmax);
    float s = simd_sum(ev);
    if (lane == 0) tg_buf[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = tg_buf[lane];
        v = simd_sum(v);
        if (lane == 0) row_sum = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    y[off + t] = ev / row_sum;
}
