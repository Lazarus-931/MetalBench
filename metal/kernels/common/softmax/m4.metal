// softmax per row: 256 threads, each handles 4 contiguous elements via float4.
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
    const uint sg = t >> 5;   // 0..7
    const uint sl = t & 31;
    const uint base = row * N + t * 4;

    threadgroup float tg_buf[8];
    threadgroup float row_max;
    threadgroup float row_sum;

    float4 v = *(device const float4*)(&x[base]);

    float m = max(max(v.x, v.y), max(v.z, v.w));
    m = simd_max(m);
    if (sl == 0) tg_buf[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float mm = (sl < 8) ? tg_buf[sl] : -INFINITY;
        mm = simd_max(mm);
        if (sl == 0) row_max = mm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rmax = row_max;

    float4 e;
    e.x = fast::exp(v.x - rmax);
    e.y = fast::exp(v.y - rmax);
    e.z = fast::exp(v.z - rmax);
    e.w = fast::exp(v.w - rmax);
    float s = (e.x + e.y) + (e.z + e.w);
    s = simd_sum(s);
    if (sl == 0) tg_buf[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float ss = (sl < 8) ? tg_buf[sl] : 0.0f;
        ss = simd_sum(ss);
        if (sl == 0) row_sum = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv = 1.0f / row_sum;
    *(device float4*)(&y[base]) = e * inv;
}
