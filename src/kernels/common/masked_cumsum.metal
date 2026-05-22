// masked_cumsum: per-row cumsum(x * mask). Two-level scan.
#include <metal_stdlib>
using namespace metal;

kernel void masked_cumsum_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  m       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   C       [[buffer(3)]],
    uint3 tid                   [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint t = tid.x;
    const uint row = tgid.y;
    const uint sg = t >> 5;
    const uint sl = t & 31;

    float v = x[row * C + t] * m[row * C + t];
    float local_scan = simd_prefix_inclusive_sum(v);

    threadgroup float tg_totals[32];
    if (sl == 31) tg_totals[sg] = local_scan;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float my_offset = 0.0f;
    if (sg == 0) my_offset = simd_prefix_exclusive_sum(tg_totals[sl]);
    if (t < 32) tg_totals[t] = my_offset;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    y[row * C + t] = local_scan + tg_totals[sg];
}
