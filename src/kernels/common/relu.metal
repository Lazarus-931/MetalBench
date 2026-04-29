// relu: out = max(x, 0). 1024 thr/tg, float4 vectorized, grid-stride loop.
#include <metal_stdlib>
using namespace metal;

kernel void relu_f32(
    device const float*  x         [[buffer(0)]],
    device       float*  y         [[buffer(1)]],
    constant     uint&   N         [[buffer(2)]],
    constant     uint&   grid_size [[buffer(3)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const uint n4 = N / 4;
    for (uint i = tid; i < n4; i += grid_size) {
        float4 v = *reinterpret_cast<const device float4*>(&x[i * 4]);
        v = fmax(v, 0.0f);
        *reinterpret_cast<device float4*>(&y[i * 4]) = v;
    }
    // Tail: scalar cleanup for remaining elements.
    for (uint i = n4 * 4 + tid; i < N; i += grid_size) {
        float v = x[i];
        y[i] = fmax(v, 0.0f);
    }
}
