// add_swish: y = swish(x + a)
#include <metal_stdlib>
using namespace metal;

kernel void add_swish_f32(
    device const float* x [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device       float* y [[buffer(2)]],
    constant uint& N      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    float s = x[tid] + a[tid];
    y[tid] = s * (1.0f / (1.0f + exp(-s)));
}
