// cumprod: float4-per-thread 2-level prefix product.
// 256 threads/TG (8 simdgroups), each handles 4 contiguous elements via float4 load/store.
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
    const uint sg = t >> 5;        // 0..7
    const uint sl = t & 31;

    const uint base = row * N + t * 4;
    float4 v = *reinterpret_cast<const device float4*>(&x[base]);

    // Local cumulative product within float4.
    float4 lp;
    lp.x = v.x;
    lp.y = lp.x * v.y;
    lp.z = lp.y * v.z;
    lp.w = lp.z * v.w;

    // simd_prefix_inclusive over float4 totals.
    float local_scan = simd_prefix_inclusive_product(lp.w);

    // Cross-simd: 8 simdgroups in TG.
    threadgroup float tg_totals[8];
    if (sl == 31) tg_totals[sg] = local_scan;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float v8 = (sl < 8) ? tg_totals[sl] : 1.0f;
        float pre = simd_prefix_exclusive_product(v8);
        if (sl < 8) tg_totals[sl] = pre;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float thread_excl_in_simd = local_scan / lp.w;
    float thread_offset = tg_totals[sg] * thread_excl_in_simd;

    float4 out;
    out.x = thread_offset * lp.x;
    out.y = thread_offset * lp.y;
    out.z = thread_offset * lp.z;
    out.w = thread_offset * lp.w;

    *reinterpret_cast<device float4*>(&y[base]) = out;
}
