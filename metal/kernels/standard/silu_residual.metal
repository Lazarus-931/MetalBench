// silu_residual: y = x + silu(a)
#include <metal_stdlib>
using namespace metal;

kernel void silu_residual_f32(
    device const float* x [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device       float* y [[buffer(2)]],
    constant uint& N      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    float xv = x[tid], av = a[tid];
    y[tid] = xv + av * (1.0f / (1.0f + exp(-av)));
}
