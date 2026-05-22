// log_softmax: y = x - logsumexp(x). One TG per row. C == TG == 1024.
#include <metal_stdlib>
using namespace metal;

kernel void log_softmax_f32(
    device const float*  X       [[buffer(0)]],
    device       float*  Y       [[buffer(1)]],
    constant     uint&   C       [[buffer(2)]],
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint t = tid3.x;
    const uint row = tgid.y;
    const uint off = row * C;
    const uint lane = t & 31u;
    const uint sg = t >> 5;

    threadgroup float tg_buf[32];

    float v = X[off + t];

    // max
    float m = simd_max(v);
    if (lane == 0) tg_buf[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rmax = simd_max(tg_buf[lane]);

    // exp + sum
    float ev = precise::exp(v - rmax);
    float s = simd_sum(ev);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) tg_buf[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rsum = simd_sum(tg_buf[lane]);

    float lse = rmax + log(rsum);
    Y[off + t] = v - lse;
}
