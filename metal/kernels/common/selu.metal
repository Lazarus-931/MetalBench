// selu: scale * (max(x,0) + alpha * min(exp(x)-1, 0)).
#include <metal_stdlib>
using namespace metal;

constant constexpr float selu_scale = 1.05070098f;
constant constexpr float selu_alpha = 1.67326324f;

kernel void selu_f32(
    device const float*  x         [[buffer(0)]],
    device       float*  y         [[buffer(1)]],
    constant     uint&   N         [[buffer(2)]],
    constant     uint&   grid_size [[buffer(3)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const uint n4 = N / 4;
    for (uint i = tid; i < n4; i += grid_size) {
        float4 v = *(reinterpret_cast<const device float4*>(&x[i * 4]));
        v = selu_scale * (fmax(v, 0.0f) + selu_alpha * fmin(exp(v) - 1.0f, 0.0f));
        *(reinterpret_cast<device float4*>(&y[i * 4])) = v;
    }
    for (uint i = n4 * 4 + tid; i < N; i += grid_size) {
        float v = x[i];
        y[i] = selu_scale * (fmax(v, 0.0f) + selu_alpha * fmin(exp(v) - 1.0f, 0.0f));
    }
}
