// cumprod: 2-level prefix product using hardware simd_prefix_inclusive_product.
// 2 threadgroup barriers.
#include <metal_stdlib>
using namespace metal;

kernel void cumprod_f32(
    device const float*  x  [[buffer(0)]],
    device       float*  y  [[buffer(1)]],
    constant     uint&   N  [[buffer(2)]],
    uint3 tid              [[thread_position_in_threadgroup]],
    uint3 tgid             [[threadgroup_position_in_grid]])
{
    const uint t = tid.x;
    const uint row = tgid.y;
    const uint sg = t >> 5;
    const uint sl = t & 31;

    float val = x[row * N + t];

    // Level 1: simdgroup prefix product
    float local_scan = simd_prefix_inclusive_product(val);

    // Last lane holds group total
    threadgroup float tg_totals[32];
    if (sl == 31) tg_totals[sg] = local_scan;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Level 2: exclusive scan over group totals
    float my_offset = 0.0f;
    if (sg == 0) {
        // For product scan, we need 1.0 (identity) as the starting point
        my_offset = simd_prefix_exclusive_product(tg_totals[sl]);
    }
    if (t < 32) tg_totals[t] = my_offset;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Add offset -> multiply offset
    y[row * N + t] = local_scan * tg_totals[sg];
}
