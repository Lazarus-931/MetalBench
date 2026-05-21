// elu: x if x>0 else α*(exp(x)-1). 1024 thr/tg, float4, grid-stride.
#include <metal_stdlib>
using namespace metal;

kernel void elu_f32(
    device const float*  x         [[buffer(0)]],
    device       float*  y         [[buffer(1)]],
    constant     uint&   N         [[buffer(2)]],
    constant     uint&   grid_size [[buffer(3)]],
    constant     float&  alpha     [[buffer(4)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const uint n4 = N / 4;
    for (uint i = tid; i < n4; i += grid_size) {
        float4 v = *reinterpret_cast<const device float4*>(&x[i * 4]);
        float4 neg = alpha * (exp(v) - 1.0f);
        float4 out = select(neg, v, v > 0.0f);
        *reinterpret_cast<device float4*>(&y[i * 4]) = out;
    }
    for (uint i = n4 * 4 + tid; i < N; i += grid_size) {
        float v = x[i];
        y[i] = v > 0.0f ? v : alpha * (exp(v) - 1.0f);
    }
}
