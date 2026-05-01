// softmax per row: 3-barrier fused reduction.
// Barrier 1: partial max → global max. Barrier 2: partial sum → global sum.
// exp computed between barriers 1 and 2. No extra broadcast barriers.
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
    float val = x[off + t];

    // Max reduction: simd_max → store → barrier → reduce in first 32 threads
    threadgroup float tg_buf[32];
    float sg_val = simd_max(val);
    if ((t & 31) == 0) tg_buf[sg] = sg_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t < 32) {
        sg_val = tg_buf[t];
        for (uint s = 16; s > 0; s >>= 1)
            sg_val = max(sg_val, simd_shuffle_down(sg_val, s));
    }
    // Broadcast via tg_buf — all threads need rmax for exp
    if (t == 0) tg_buf[0] = sg_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rmax = tg_buf[0];

    // Exp + sum: compute exp, simd_sum, store → barrier → reduce
    float exp_val = exp(val - rmax);
    sg_val = simd_sum(exp_val);
    if ((t & 31) == 0) tg_buf[sg] = sg_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t < 32) {
        sg_val = tg_buf[t];
        for (uint s = 16; s > 0; s >>= 1)
            sg_val += simd_shuffle_down(sg_val, s);
    }
    if (t == 0) tg_buf[0] = sg_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    y[off + t] = exp_val / tg_buf[0];
}
