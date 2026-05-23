// add_norm M4: y = layer_norm(x + residual). D=1024, TG=1024.
// Single-simdgroup-per-row: only sg 0 (32 lanes) does the work; each lane
// processes 8 float4 = 32 elements -> covers the full 1024-wide row.
// No threadgroup memory, no barriers: a pair of simd_sum() does it all.
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
    // Only the first simdgroup participates.
    if (t >= 32u) return;

    const uint row = tgid.y;
    const uint D4  = D >> 2;            // 256
    const uint per = D4 >> 5;           // 8 float4 per lane

    device const float4* xr = (device const float4*)(x + row * D);
    device const float4* rr = (device const float4*)(res + row * D);
    device       float4* yr = (device       float4*)(y + row * D);

    // Load and add in one pass; cache for the second pass.
    float4 cache[8];
    float ssum = 0.0f, ssq = 0.0f;
    #pragma unroll
    for (uint k = 0; k < 8; ++k) {
        uint i = k * 32u + t;            // strided so consecutive lanes hit consecutive float4
        float4 u = xr[i] + rr[i];
        cache[k] = u;
        ssum += u.x + u.y + u.z + u.w;
        ssq  += dot(u, u);
    }

    ssum = simd_sum(ssum);
    ssq  = simd_sum(ssq);

    const float invD = 1.0f / float(D);
    const float mean = ssum * invD;
    const float var  = max(ssq * invD - mean * mean, 0.0f);
    const float inv_std = rsqrt(var + eps);

    #pragma unroll
    for (uint k = 0; k < 8; ++k) {
        uint i = k * 32u + t;
        yr[i] = (cache[k] - mean) * inv_std;
    }
    (void)per;
}
