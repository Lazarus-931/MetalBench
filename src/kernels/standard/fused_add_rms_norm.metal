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
    device const float* xr = X + row * D;
    device const float* rr = R + row * D;
    device       float* yr = Y + row * D;

    threadgroup float reduce[32];

    float sumsq = 0.0f;
    for (uint i = tid; i < D; i += TG) {
        float v = xr[i] + rr[i];
        sumsq += v * v;
    }
    sumsq = simd_sum(sumsq);
    if ((tid & 31) == 0) reduce[tid >> 5] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        sumsq = reduce[tid];
        for (uint s = 16; s > 0; s >>= 1) sumsq += simd_shuffle_down(sumsq, s);
        if (tid == 0) reduce[0] = sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt(reduce[0] / float(D) + eps);
    for (uint i = tid; i < D; i += TG) yr[i] = (xr[i] + rr[i]) * inv;
}
