// residual_add: y = x + alpha * residual. Vectorized: single float4 per thread.
#include <metal_stdlib>
using namespace metal;

kernel void residual_add_f32(
    device const float*  X       [[buffer(0)]],
    device const float*  R       [[buffer(1)]],
    device       float*  Y       [[buffer(2)]],
    constant     uint&   N       [[buffer(3)]],
    constant     uint&   grid_size [[buffer(4)]],
    constant     float&  alpha   [[buffer(5)]],
    uint tid                    [[thread_position_in_grid]])
{
    const uint n4 = N >> 2;           // 1<<18
    const float a = alpha;
    for (uint i = tid; i < n4; i += grid_size) {
        uint base = i << 2;
        float4 x = *reinterpret_cast<const device float4*>(&X[base]);
        float4 r = *reinterpret_cast<const device float4*>(&R[base]);
        *reinterpret_cast<device float4*>(&Y[base]) = x + a * r;
    }
}
