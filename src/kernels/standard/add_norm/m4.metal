// add_norm M4: y = layer_norm(x + residual). D=1024, TG=1024.
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
    const uint TG = 1024;
    const uint t = tid.x;
    const uint row = tgid.y;
    const uint lane = t & 31u;
    const uint sg = t >> 5;

    const uint D4 = D >> 2;
    device const float4* xr = (device const float4*)(x + row * D);
    device const float4* rr = (device const float4*)(res + row * D);
    device       float4* yr = (device       float4*)(y + row * D);

    float4 v4 = 0.0f;
    float ssum = 0.0f, ssq = 0.0f;
    bool active = t < D4;
    if (active) {
        v4 = xr[t] + rr[t];
        ssum = v4.x + v4.y + v4.z + v4.w;
        ssq  = dot(v4, v4);
    }
    for (uint i = t + TG; i < D4; i += TG) {
        float4 u = xr[i] + rr[i];
        ssum += u.x + u.y + u.z + u.w;
        ssq  += dot(u, u);
    }

    ssum = simd_sum(ssum);
    ssq  = simd_sum(ssq);

    threadgroup float2 tg_buf[32];
    if (lane == 0) tg_buf[sg] = float2(ssum, ssq);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float2 b = tg_buf[lane];
    float s_all  = simd_sum(b.x);
    float sq_all = simd_sum(b.y);

    float invD = 1.0f / float(D);
    float mean = s_all * invD;
    float var  = max(sq_all * invD - mean * mean, 0.0f);
    float inv_std = rsqrt(var + eps);

    if (active) {
        yr[t] = (v4 - mean) * inv_std;
    }
    for (uint i = t + TG; i < D4; i += TG) {
        float4 u = xr[i] + rr[i];
        yr[i] = (u - mean) * inv_std;
    }
}
