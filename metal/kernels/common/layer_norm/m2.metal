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

    // Compute sum and sumsq using fused multiply-add for sumsq to reduce instruction count
    float4 s4 = ((v0 + v1) + (v2 + v3)) + ((v4 + v5) + (v6 + v7));
    float sum   = (s4.x + s4.y) + (s4.z + s4.w);
    float sumsq = fma(v0.x, v0.x, fma(v0.y, v0.y, fma(v0.z, v0.z, fma(v0.w, v0.w,
                  fma(v1.x, v1.x, fma(v1.y, v1.y, fma(v1.z, v1.z, fma(v1.w, v1.w,
                  fma(v2.x, v2.x, fma(v2.y, v2.y, fma(v2.z, v2.z, fma(v2.w, v2.w,
                  fma(v3.x, v3.x, fma(v3.y, v3.y, fma(v3.z, v3.z, fma(v3.w, v3.w,
                  fma(v4.x, v4.x, fma(v4.y, v4.y, fma(v4.z, v4.z, fma(v4.w, v4.w,
                  fma(v5.x, v5.x, fma(v5.y, v5.y, fma(v5.z, v5.z, fma(v5.w, v5.w,
                  fma(v6.x, v6.x, fma(v6.y, v6.y, fma(v6.z, v6.z, fma(v6.w, v6.w,
                  fma(v7.x, v7.x, fma(v7.y, v7.y, fma(v7.z, v7.z, v7.w * v7.w)))))))))))))))))))))))))))))));

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
