// cumsum: 2-level prefix scan using hardware simd_prefix_inclusive_sum.
// 2 threadgroup barriers vs 20 for Hillis-Steele.
#include <metal_stdlib>
using namespace metal;

kernel void cumsum_f32(
    device const float*  x  [[buffer(0)]],
    device       float*  y  [[buffer(1)]],
    constant     uint&   N  [[buffer(2)]],
    uint3 tid              [[thread_position_in_threadgroup]],
    uint3 tgid             [[threadgroup_position_in_grid]])
{
    const uint t = tid.x;
    const uint row = tgid.y;
    const uint sg = t >> 5;        // simdgroup index (0..31)
    const uint sl = t & 31;         // lane within simdgroup (0..31)

    float val = x[row * N + t];

    // Level 1: simdgroup scan (hardware-accelerated, no barrier)
    float local_scan = simd_prefix_inclusive_sum(val);

    // Last lane in each simdgroup holds the group's total sum
    threadgroup float tg_totals[32];
    if (sl == 31) tg_totals[sg] = local_scan;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Level 2: first simdgroup computes exclusive scan over the 32 totals.
    // After this, thread sl (if sg==0) holds the offset for simdgroup sl.
    float my_offset = 0.0f;
    if (sg == 0) {
        my_offset = simd_prefix_exclusive_sum(tg_totals[sl]);
    }
    // Broadcast: thread sl writes the offset for simdgroup sl.
    if (t < 32) tg_totals[t] = my_offset;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread adds its simdgroup's offset.
    y[row * N + t] = local_scan + tg_totals[sg];
}
