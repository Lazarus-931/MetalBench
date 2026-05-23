#include <metal_stdlib>
using namespace metal;

kernel void mingpt_new_gelu_f32(
    device const float*  x         [[buffer(0)]],
    device       float*  y         [[buffer(1)]],
    constant     uint&   N         [[buffer(2)]],
    constant     uint&   grid_size [[buffer(3)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const float k0 = 0.7978845608f;
    const float k1 = 0.044715f;
    const uint n4 = N / 4;
    for (uint i = tid; i < n4; i += grid_size) {
        float4 v = *reinterpret_cast<const device float4*>(&x[i * 4]);
        float4 t = k0 * (v + k1 * v * v * v);
        *reinterpret_cast<device float4*>(&y[i * 4]) = 0.5f * v * (1.0f + precise::tanh(t));
    }
    for (uint i = n4 * 4 + tid; i < N; i += grid_size) {
        float v = x[i];
        float t = k0 * (v + k1 * v * v * v);
        y[i] = 0.5f * v * (1.0f + precise::tanh(t));
    }
}
