// logsigmoid: numerically stable log(sigmoid(x)).
#include <metal_stdlib>
using namespace metal;

inline float logsigmoid_scalar(float v) {
    if (v > 0.0f) {
        return -log(1.0f + exp(-v));
    } else {
        return v - log(1.0f + exp(v));
    }
}

kernel void logsigmoid_f32(
    device const float*  x         [[buffer(0)]],
    device       float*  y         [[buffer(1)]],
    constant     uint&   N         [[buffer(2)]],
    constant     uint&   grid_size [[buffer(3)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const uint n4 = N / 4;
    for (uint i = tid; i < n4; i += grid_size) {
        float4 v = *(reinterpret_cast<const device float4*>(&x[i * 4]));
        v = float4(
            logsigmoid_scalar(v.x),
            logsigmoid_scalar(v.y),
            logsigmoid_scalar(v.z),
            logsigmoid_scalar(v.w));
        *(reinterpret_cast<device float4*>(&y[i * 4])) = v;
    }
    for (uint i = n4 * 4 + tid; i < N; i += grid_size) {
        y[i] = logsigmoid_scalar(x[i]);
    }
}
