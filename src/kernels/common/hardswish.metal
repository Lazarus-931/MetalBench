// hardswish: x * clamp(x+3, 0, 6) / 6. Float4 grid-stride.
#include <metal_stdlib>
using namespace metal;

kernel void hardswish_f32(
    device const float*  x         [[buffer(0)]],
    device       float*  y         [[buffer(1)]],
    constant     uint&   N         [[buffer(2)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const uint grid_size = 64 * 1024;
    const uint n4 = N / 4;
    for (uint i = tid; i < n4; i += grid_size) {
        float4 v = *(reinterpret_cast<const device float4*>(&x[i * 4]));
        *(reinterpret_cast<device float4*>(&y[i * 4])) = v * clamp(v + 3.0f, 0.0f, 6.0f) / 6.0f;
    }
    for (uint i = n4 * 4 + tid; i < N; i += grid_size)
        y[i] = x[i] * clamp(x[i] + 3.0f, 0.0f, 6.0f) / 6.0f;
}
