// fused_add_rms_norm: y = (x+r) * rsqrt(mean((x+r)^2) + eps). Per-row.
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
    const uint TG = 1024;
    const uint tid = tid3.x;
    const uint row = tgid.y;

    // D is 1024 → 256 float4s per row. With TG=1024 threads, only 256 do work.
    // We still must keep TG=1024 because registry says so.
    device const float4* xr = (device const float4*)(X + row * D);
    device const float4* rr = (device const float4*)(R + row * D);
    device       float4* yr = (device       float4*)(Y + row * D);

    const uint D4 = D >> 2;

    threadgroup float reduce[32];

    float sumsq = 0.0f;
    float4 cached;
    bool active = tid < D4;
    if (active) {
        cached = xr[tid] + rr[tid];
        sumsq = dot(cached, cached);
    }
    // Handle case D4 > TG (general): unroll grid-stride
    for (uint i = tid + TG; i < D4; i += TG) {
        float4 v = xr[i] + rr[i];
        sumsq += dot(v, v);
    }

    sumsq = simd_sum(sumsq);
    if ((tid & 31) == 0) reduce[tid >> 5] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = reduce[tid];
        v = simd_sum(v);
        if (tid == 0) reduce[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt(reduce[0] / float(D) + eps);

    if (active) {
        yr[tid] = cached * inv;
    }
    for (uint i = tid + TG; i < D4; i += TG) {
        float4 v = xr[i] + rr[i];
        yr[i] = v * inv;
    }
}
