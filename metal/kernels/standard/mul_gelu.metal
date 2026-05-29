// mul_gelu: y = gelu(x * a)
#include <metal_stdlib>
using namespace metal;

kernel void mul_gelu_f32(
    device const float* x [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device       float* y [[buffer(2)]],
    constant uint& N      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    float s = x[tid] * a[tid];
    y[tid] = 0.5f * s * (1.0f + tanh(0.7978845608f * (s + 0.044715f * s*s*s)));
}
