// gelu: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
#include <metal_stdlib>
using namespace metal;

constant constexpr float gelu_k = 0.79788456f;

kernel void gelu_f32(
    device const float*  x         [[buffer(0)]],
    device       float*  y         [[buffer(1)]],
    constant     uint&   N         [[buffer(2)]],
    constant     uint&   grid_size [[buffer(3)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const uint n4 = N / 4;
    for (uint i = tid; i < n4; i += grid_size) {
        float4 v = *(reinterpret_cast<const device float4*>(&x[i * 4]));
        v = 0.5f * v * (1.0f + tanh(gelu_k * (v + 0.044715f * v * v * v)));
        *(reinterpret_cast<device float4*>(&y[i * 4])) = v;
    }
    for (uint i = n4 * 4 + tid; i < N; i += grid_size) {
        float v = x[i];
        y[i] = 0.5f * v * (1.0f + tanh(gelu_k * (v + 0.044715f * v * v * v)));
    }
}
