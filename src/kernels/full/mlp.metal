// mlp: x @ W1 → GELU → @ W2 → GELU → @ W3. Single TG, sequential layers via TG memory.
// Shapes: x (16, 128), W1 (128, 512), W2 (512, 128), W3 (128, 10).
#include <metal_stdlib>
using namespace metal;

static inline float gelu_tanh(float v) {
    float t = 0.7978845608f * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + precise::tanh(t));
}

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
    threadgroup float h1[16 * 512];
    threadgroup float h2[16 * 128];

    for (uint i = tid; i < N * D2; i += 1024) {
        uint r = i / D2, c = i % D2;
        float s = 0.0f;
        for (uint k = 0; k < D1; ++k) s += x[r * D1 + k] * W1[k * D2 + c];
        h1[r * D2 + c] = gelu_tanh(s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < N * D1; i += 1024) {
        uint r = i / D1, c = i % D1;
        float s = 0.0f;
        for (uint k = 0; k < D2; ++k) s += h1[r * D2 + k] * W2[k * D1 + c];
        h2[r * D1 + c] = gelu_tanh(s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < N * Do; i += 1024) {
        uint r = i / Do, c = i % Do;
        float s = 0.0f;
        for (uint k = 0; k < D1; ++k) s += h2[r * D1 + k] * W3[k * Do + c];
        y[r * Do + c] = s;
    }
}
