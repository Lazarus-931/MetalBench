// add_norm: y = layer_norm(x + residual). Fused add + norm in one pass.
#include <metal_stdlib>
using namespace metal;

kernel void add_norm_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  res     [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   D       [[buffer(3)]],
    constant     float&  eps     [[buffer(4)]],
    uint3 tid                   [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint t = tid.x;
    const uint row = tgid.y;
    const uint off = row * D;

    float val = x[off + t] + res[off + t]; // Fused add
    float s2  = val * val;

    float sum   = simd_sum(val);
    float sumsq = simd_sum(s2);

    threadgroup float tg_sum[32], tg_sumsq[32];
    const uint sg = t >> 5;
    if ((t & 31) == 0) { tg_sum[sg] = sum; tg_sumsq[sg] = sumsq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t < 32) {
        sum = tg_sum[t]; sumsq = tg_sumsq[t];
        for (uint s = 16; s > 0; s >>= 1) {
            sum   += simd_shuffle_down(sum,   s);
            sumsq += simd_shuffle_down(sumsq, s);
        }
    }
    if (t == 0) { tg_sum[0] = sum; tg_sumsq[0] = sumsq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = tg_sum[0] / float(D);
    float var  = max(tg_sumsq[0] / float(D) - mean * mean, 0.0f);
    float inv_std = rsqrt(var + eps);

    y[off + t] = (val - mean) * inv_std;
}
