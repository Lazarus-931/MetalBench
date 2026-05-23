// sigmoid: out = 1 / (1 + fast::exp(-x)). 1024 thr/tg, float4, grid-stride.
#include <metal_stdlib>
using namespace metal;

kernel void sigmoid_f32(
    device const float*  x         [[buffer(0)]],
    device       float*  y         [[buffer(1)]],
    constant     uint&   N         [[buffer(2)]],
    constant     uint&   grid_size [[buffer(3)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const uint n4 = N / 4;
    for (uint i = tid; i < n4; i += grid_size) {
        float4 v = *reinterpret_cast<const device float4*>(&x[i * 4]);
        v = 1.0f / (1.0f + fast::exp(-v));
        *reinterpret_cast<device float4*>(&y[i * 4]) = v;
    }
    for (uint i = n4 * 4 + tid; i < N; i += grid_size) {
        float v = x[i];
        y[i] = 1.0f / (1.0f + fast::exp(-v));
    }
}
