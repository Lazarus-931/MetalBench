// cumsum_exclusive: per-row exclusive prefix sum. Two-level scan via simd hardware.
#include <metal_stdlib>
using namespace metal;

kernel void cumsum_exclusive_f32(
    device const float*  x       [[buffer(0)]],
    device       float*  y       [[buffer(1)]],
    constant     uint&   C       [[buffer(2)]],
    uint3 tid                   [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint t = tid.x;
    const uint row = tgid.y;
    const uint sg = t >> 5;
    const uint sl = t & 31;

    float v = x[row * C + t];
    float local_scan = simd_prefix_inclusive_sum(v);

    threadgroup float tg_totals[32];
    if (sl == 31) tg_totals[sg] = local_scan;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float my_offset = 0.0f;
    if (sg == 0) my_offset = simd_prefix_exclusive_sum(tg_totals[sl]);
    if (t < 32) tg_totals[t] = my_offset;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inclusive total at this thread, then subtract own v to get exclusive.
    y[row * C + t] = local_scan + tg_totals[sg] - v;
}
