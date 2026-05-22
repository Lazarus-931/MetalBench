// layer_norm: y = (x - mean) * rsqrt(var + eps) per row.
#include <metal_stdlib>
using namespace metal;

kernel void layer_norm_f32(
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
    const uint lane = t & 31u;
    const uint sg = t >> 5;

    float val = x[off + t];
    float sum   = simd_sum(val);
    float sumsq = simd_sum(val * val);

    threadgroup float2 tg_buf[32];
    if (lane == 0) tg_buf[sg] = float2(sum, sumsq);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float2 v = tg_buf[lane];
    float s_all  = simd_sum(v.x);
    float sq_all = simd_sum(v.y);

    float invN = 1.0f / float(N);
    float mean = s_all * invN;
    float var  = max(sq_all * invN - mean * mean, 0.0f);
    float inv_std = rsqrt(var + eps);

    y[off + t] = (val - mean) * inv_std;
}
