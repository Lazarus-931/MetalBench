// masked_cumsum: per-row cumsum(x * mask). Two-level scan with float4 loads.
// 1024-col row -> 256 active threads * float4 = 1024 elements per TG.
#include <metal_stdlib>
using namespace metal;

kernel void masked_cumsum_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  m       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   C       [[buffer(3)]],
    uint3 tid                    [[thread_position_in_threadgroup]],
    uint3 tgid                   [[threadgroup_position_in_grid]])
{
    constexpr uint NSG = 8;        // 256/32
    threadgroup float tg_totals[NSG];

    const uint t   = tid.x;
    const uint row = tgid.y;
    const uint sg  = t >> 5;
    const uint sl  = t & 31;
    const bool active = (t < 256u);

    float4 s = float4(0.0f);
    float local_inc = 0.0f;
    float local_exc = 0.0f;
    uint base = 0u;

    if (active) {
        base = row * C + (t << 2);
        // Issue both loads early for ILP
        float4 xv = *reinterpret_cast<device const float4*>(x + base);
        float4 mv = *reinterpret_cast<device const float4*>(m + base);
        float4 v = xv * mv;
        s.x = v.x;
        s.y = s.x + v.y;
        s.z = s.y + v.z;
        s.w = s.z + v.w;
        local_inc = simd_prefix_inclusive_sum(s.w);
        local_exc = local_inc - s.w;

        if (sl == 31u) tg_totals[sg] = local_inc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        float v8 = (sl < NSG) ? tg_totals[sl] : 0.0f;
        float prefix = simd_prefix_exclusive_sum(v8);
        if (sl < NSG) tg_totals[sl] = prefix;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
        float off = tg_totals[sg] + local_exc;
        float4 out = s + off;
        *reinterpret_cast<device float4*>(y + base) = out;
    }
}
