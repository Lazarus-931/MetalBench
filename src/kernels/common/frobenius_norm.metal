// frobenius_norm: ||x||_F = sqrt(sum(x*x)). Single TG, 1024 threads. N=1024*1024.
// float4 loads to better saturate bandwidth.
#include <metal_stdlib>
using namespace metal;

kernel void frobenius_norm_f32(
    device const float*  x       [[buffer(0)]],
    device       float*  y       [[buffer(1)]],
    constant     uint&   N       [[buffer(2)]],
    uint3 tid3                  [[thread_position_in_threadgroup]])
{
    const uint TG = 1024;
    const uint tid = tid3.x;
    threadgroup float reduce[32];

    device const float4* x4 = (device const float4*)x;
    const uint N4 = N >> 2;

    float s = 0.0f;
    for (uint i = tid; i < N4; i += TG) {
        float4 v = x4[i];
        s += dot(v, v);
    }
    s = simd_sum(s);
    if ((tid & 31) == 0) reduce[tid >> 5] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        s = reduce[tid];
        s = simd_sum(s);
        if (tid == 0) y[0] = sqrt(s);
    }
}
