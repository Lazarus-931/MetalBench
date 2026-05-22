// fused_add_rms_norm: y = (x+r) * rsqrt(mean((x+r)^2) + eps). Per-row.
// D=1024, TG=1024. Single-simdgroup-per-row: only sg 0 (32 lanes) works,
// each lane handles 8 float4 = 32 elements. No threadgroup memory, no
// barriers; just a single simd_sum() for the reduction.
#include <metal_stdlib>
using namespace metal;

kernel void fused_add_rms_norm_f32(
    device const float*  X       [[buffer(0)]],
    device const float*  R       [[buffer(1)]],
    device       float*  Y       [[buffer(2)]],
    constant     uint&   D       [[buffer(3)]],
    constant     float&  eps     [[buffer(4)]],
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint t = tid3.x;
    if (t >= 32u) return;

    const uint row = tgid.y;

    device const float4* xr = (device const float4*)(X + row * D);
    device const float4* rr = (device const float4*)(R + row * D);
    device       float4* yr = (device       float4*)(Y + row * D);

    float4 cache[8];
    float sumsq = 0.0f;
    #pragma unroll
    for (uint k = 0; k < 8; ++k) {
        uint i = k * 32u + t;
        float4 v = xr[i] + rr[i];
        cache[k] = v;
        sumsq += dot(v, v);
    }

    sumsq = simd_sum(sumsq);
    float inv = rsqrt(sumsq / float(D) + eps);

    #pragma unroll
    for (uint k = 0; k < 8; ++k) {
        uint i = k * 32u + t;
        yr[i] = cache[k] * inv;
    }
}
