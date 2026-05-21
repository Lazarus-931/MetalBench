// dropout (inverted): y = mask * x * (1/(1-p)). float4 grid-stride.
#include <metal_stdlib>
using namespace metal;

kernel void dropout_f32(
    device const float*  X       [[buffer(0)]],
    device const float*  M       [[buffer(1)]],
    device       float*  Y       [[buffer(2)]],
    constant     uint&   N       [[buffer(3)]],
    constant     uint&   grid_size [[buffer(4)]],
    constant     float&  p       [[buffer(5)]],
    uint tid                    [[thread_position_in_grid]])
{
    const float scale = 1.0f / (1.0f - p);
    const uint n4 = N >> 2;
    for (uint i = tid; i < n4; i += grid_size) {
        uint base = i << 2;
        float4 x = *reinterpret_cast<const device float4*>(&X[base]);
        float4 m = *reinterpret_cast<const device float4*>(&M[base]);
        *reinterpret_cast<device float4*>(&Y[base]) = m * x * scale;
    }
}
