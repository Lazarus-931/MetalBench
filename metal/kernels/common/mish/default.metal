// mish: x * tanh(softplus(x)). Stable softplus = max(x,0) + log1p(exp(-|x|)).
#include <metal_stdlib>
using namespace metal;

static inline float4 softplus4(float4 v) {
    float4 ax = fabs(v);
    float4 mx = fmax(v, 0.0f);
    return mx + log(1.0f + exp(-ax));
}

kernel void mish_f32(
    device const float*  x         [[buffer(0)]],
    device       float*  y         [[buffer(1)]],
    constant     uint&   N         [[buffer(2)]],
    constant     uint&   grid_size [[buffer(3)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const uint n4 = N / 4;
    for (uint i = tid; i < n4; i += grid_size) {
        float4 v = *reinterpret_cast<const device float4*>(&x[i * 4]);
        float4 sp = softplus4(v);
        *reinterpret_cast<device float4*>(&y[i * 4]) = v * tanh(sp);
    }
    for (uint i = n4 * 4 + tid; i < N; i += grid_size) {
        float v = x[i];
        float sp = fmax(v, 0.0f) + log(1.0f + exp(-fabs(v)));
        y[i] = v * tanh(sp);
    }
}
