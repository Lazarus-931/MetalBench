// gelu_residual: y = x + gelu(a) using tanh approximation
#include <metal_stdlib>
using namespace metal;

kernel void gelu_residual_f32(
    device const float* x [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device       float* y [[buffer(2)]],
    constant uint& N      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    float xv = x[tid], av = a[tid];
    float g = 0.5f * av * (1.0f + tanh(0.7978845608f * (av + 0.044715f * av*av*av)));
    y[tid] = xv + g;
}
