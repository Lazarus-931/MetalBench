// mul_relu: y = relu(x * a)
#include <metal_stdlib>
using namespace metal;

kernel void mul_relu_f32(
    device const float* x [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device       float* y [[buffer(2)]],
    constant uint& N      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    y[tid] = max(x[tid] * a[tid], 0.0f);
}
