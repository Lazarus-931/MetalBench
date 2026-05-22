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
    const uint lane = t & 31u;
    const uint sg = t >> 5;

    float val = x[off + t] + res[off + t];

    float sum   = simd_sum(val);
    float sumsq = simd_sum(val * val);

    threadgroup float2 tg_buf[32];
    if (lane == 0) tg_buf[sg] = float2(sum, sumsq);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All simdgroups load and reduce — saves one broadcast barrier.
    float2 v = tg_buf[lane];
    float s_all  = simd_sum(v.x);
    float sq_all = simd_sum(v.y);

    float invD = 1.0f / float(D);
    float mean = s_all * invD;
    float var  = max(sq_all * invD - mean * mean, 0.0f);
    float inv_std = rsqrt(var + eps);

    y[off + t] = (val - mean) * inv_std;
}
