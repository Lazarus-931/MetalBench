// swish: out = x * sigmoid(x) = x / (1 + exp(-x)). 1024 thr/tg, float4.
#include <metal_stdlib>
using namespace metal;

kernel void swish_f32(
    device const float*  x         [[buffer(0)]],
    device       float*  y         [[buffer(1)]],
    constant     uint&   N         [[buffer(2)]],
    constant     uint&   grid_size [[buffer(3)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const uint n4 = N / 4;
    for (uint i = tid; i < n4; i += grid_size) {
        float4 v = *reinterpret_cast<const device float4*>(&x[i * 4]);
        v = v / (1.0f + exp(-v));
        *reinterpret_cast<device float4*>(&y[i * 4]) = v;
    }
    for (uint i = n4 * 4 + tid; i < N; i += grid_size) {
        float v = x[i];
        y[i] = v / (1.0f + exp(-v));
    }
}
