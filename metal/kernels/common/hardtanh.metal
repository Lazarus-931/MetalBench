// hardtanh: out = clamp(x, -1, 1). float4 grid-stride.
#include <metal_stdlib>
using namespace metal;

kernel void hardtanh_f32(
    device const float*  x         [[buffer(0)]],
    device       float*  y         [[buffer(1)]],
    constant     uint&   N         [[buffer(2)]],
    constant     uint&   grid_size [[buffer(3)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const uint n4 = N / 4;
    for (uint i = tid; i < n4; i += grid_size) {
        float4 v = *reinterpret_cast<const device float4*>(&x[i * 4]);
        *reinterpret_cast<device float4*>(&y[i * 4]) = clamp(v, -1.0f, 1.0f);
    }
    for (uint i = n4 * 4 + tid; i < N; i += grid_size) {
        y[i] = clamp(x[i], -1.0f, 1.0f);
    }
}
