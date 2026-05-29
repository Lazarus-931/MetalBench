// rms_norm: y = x * rsqrt(mean(x^2) + eps). Simdgroup reduction, no mean.
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_f32(
    device const float*  x       [[buffer(0)]],
    device       float*  y       [[buffer(1)]],
    constant     uint&   N       [[buffer(2)]],
    constant     float&  eps     [[buffer(3)]],
    uint3 tid                   [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint t = tid.x;
    const uint row = tgid.y;
    const uint off = row * N;
    float val = x[off + t];
    float sumsq = simd_sum(val * val);

    threadgroup float tg_sum[32];
    const uint sg = t >> 5;
    if ((t & 31) == 0) tg_sum[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t < 32) {
        sumsq = tg_sum[t];
        for (uint s = 16; s > 0; s >>= 1)
            sumsq += simd_shuffle_down(sumsq, s);
    }
    if (t == 0) tg_sum[0] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = rsqrt(tg_sum[0] / float(N) + eps);
    y[off + t] = val * inv_rms;
}
