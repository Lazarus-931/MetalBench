// masked_softmax M4: y = softmax(x + mask). Per-row. C=TG=1024.
#include <metal_stdlib>
using namespace metal;

kernel void masked_softmax_f32(
    device const float*  X       [[buffer(0)]],
    device const float*  M       [[buffer(1)]],
    device       float*  Y       [[buffer(2)]],
    constant     uint&   C       [[buffer(3)]],
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint t = tid3.x;
    const uint row = tgid.y;
    const uint off = row * C;
    const uint lane = t & 31u;
    const uint sg = t >> 5;

    threadgroup float tg_max[32];
    threadgroup float tg_sum[32];

    float val = X[off + t] + M[off + t];

    float m = simd_max(val);
    if (lane == 0) tg_max[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rmax = simd_max(tg_max[lane]);

    float ev = precise::exp(val - rmax);
    float s = simd_sum(ev);
    if (lane == 0) tg_sum[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rsum = simd_sum(tg_sum[lane]);

    Y[off + t] = ev * (1.0f / rsum);
}
