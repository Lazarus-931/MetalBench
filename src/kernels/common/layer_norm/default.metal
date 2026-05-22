// layer_norm: y = (x - mean) * rsqrt(var + eps) per row.
// D=1024, TG=1024 (registry). Single-simdgroup-per-row: only sg 0 (32 lanes)
// works, each lane handles 8 float4 = 32 elements. Cache float4 strips in
// registers across reduction and normalize passes.
#include <metal_stdlib>
using namespace metal;

kernel void layer_norm_f32(
    device const float*  X       [[buffer(0)]],
    device       float*  Y       [[buffer(1)]],
    constant     uint&   D       [[buffer(2)]],
    constant     float&  eps     [[buffer(3)]],
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint t = tid3.x;
    if (t >= 32u) return;

    const uint row = tgid.y;
    device const float4* xr = (device const float4*)(X + row * D);
    device       float4* yr = (device       float4*)(Y + row * D);

    // Each of 32 lanes handles 8 float4s (= 32 floats) -> 1024 floats total.
    float4 v0, v1, v2, v3, v4, v5, v6, v7;
    v0 = xr[0u  * 32u + t];
    v1 = xr[1u  * 32u + t];
    v2 = xr[2u  * 32u + t];
    v3 = xr[3u  * 32u + t];
    v4 = xr[4u  * 32u + t];
    v5 = xr[5u  * 32u + t];
    v6 = xr[6u  * 32u + t];
    v7 = xr[7u  * 32u + t];

    float4 s4 = ((v0 + v1) + (v2 + v3)) + ((v4 + v5) + (v6 + v7));
    float sum   = (s4.x + s4.y) + (s4.z + s4.w);
    float sumsq = dot(v0, v0) + dot(v1, v1) + dot(v2, v2) + dot(v3, v3)
                + dot(v4, v4) + dot(v5, v5) + dot(v6, v6) + dot(v7, v7);

    sum   = simd_sum(sum);
    sumsq = simd_sum(sumsq);

    float invD = 1.0f / float(D);
    float mean = sum * invD;
    float var  = max(sumsq * invD - mean * mean, 0.0f);
    float inv_std = fast::rsqrt(var + eps);

    yr[0u  * 32u + t] = (v0 - mean) * inv_std;
    yr[1u  * 32u + t] = (v1 - mean) * inv_std;
    yr[2u  * 32u + t] = (v2 - mean) * inv_std;
    yr[3u  * 32u + t] = (v3 - mean) * inv_std;
    yr[4u  * 32u + t] = (v4 - mean) * inv_std;
    yr[5u  * 32u + t] = (v5 - mean) * inv_std;
    yr[6u  * 32u + t] = (v6 - mean) * inv_std;
    yr[7u  * 32u + t] = (v7 - mean) * inv_std;
}
