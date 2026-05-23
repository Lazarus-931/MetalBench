// hardsigmoid: clamp(x/6 + 0.5, 0, 1).
#include <metal_stdlib>
using namespace metal;

kernel void hardsigmoid_f32(
    device const float*  x         [[buffer(0)]],
    device       float*  y         [[buffer(1)]],
    constant     uint&   N         [[buffer(2)]],
    constant     uint&   grid_size [[buffer(3)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const uint n4 = N / 4;
    for (uint i = tid; i < n4; i += grid_size) {
        float4 v = *(reinterpret_cast<const device float4*>(&x[i * 4]));
        v = clamp(v / 6.0f + 0.5f, 0.0f, 1.0f);
        *(reinterpret_cast<device float4*>(&y[i * 4])) = v;
    }
    for (uint i = n4 * 4 + tid; i < N; i += grid_size) {
        y[i] = clamp(x[i] / 6.0f + 0.5f, 0.0f, 1.0f);
    }
}
