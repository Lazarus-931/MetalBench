// mlp on M4: process 8 rows in parallel (half batch at a time). Fits in TG mem.
#include <metal_stdlib>
using namespace metal;

static inline float erf_approx(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-ax * ax);
    return sign * y;
}
static inline float gelu_exact(float v) {
    return 0.5f * v * (1.0f + erf_approx(v * 0.70710678118f));
}

constant constexpr uint RB = 8;

kernel void mlp_f32(
    device const float* x   [[buffer(0)]],
    device const float* W1  [[buffer(1)]],
    device const float* W2  [[buffer(2)]],
    device const float* W3  [[buffer(3)]],
    device       float* y   [[buffer(4)]],
    constant     uint& N    [[buffer(5)]],
    constant     uint& D1   [[buffer(6)]],
    constant     uint& D2   [[buffer(7)]],
    constant     uint& Do   [[buffer(8)]],
    uint3 tid3              [[thread_position_in_threadgroup]])
{
    const uint tid = tid3.x;
    threadgroup float xs[RB * 128];   // RB rows of x: 1024 floats = 4KB
    threadgroup float h1[RB * 512];   // 4096 floats = 16KB
    threadgroup float h2[RB * 128];   // 1024 floats = 4KB

    for (uint rb = 0; rb < N; rb += RB) {
        for (uint i = tid; i < RB * D1; i += 1024) {
            xs[i] = x[(rb + i / D1) * D1 + (i % D1)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < RB * D2; i += 1024) {
            uint r = i / D2;
            uint c = i % D2;
            float s = 0.0f;
            for (uint k = 0; k < D1; ++k) s += xs[r * D1 + k] * W1[k * D2 + c];
            h1[i] = gelu_exact(s);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < RB * D1; i += 1024) {
            uint r = i / D1;
            uint c = i % D1;
            float s = 0.0f;
            for (uint k = 0; k < D2; ++k) s += h1[r * D2 + k] * W2[k * D1 + c];
            h2[i] = gelu_exact(s);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < RB * Do; i += 1024) {
            uint r = i / Do;
            uint c = i % Do;
            float s = 0.0f;
            for (uint k = 0; k < D1; ++k) s += h2[r * D1 + k] * W3[k * Do + c];
            y[(rb + r) * Do + c] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
