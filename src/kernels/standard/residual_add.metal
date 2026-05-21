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
    // N = 1024*1024 = 1<<20. grid_size = 64*1024 = 1<<16.
    // Each thread does N/grid_size = 16 floats = 4 float4s.
    const uint n4 = N >> 2;           // 1<<18
    const float a = alpha;
    // Use base = tid*4 then stride grid_size in float4 units.
    for (uint i = tid; i < n4; i += grid_size) {
        uint base = i << 2;
        float4 x = *reinterpret_cast<const device float4*>(&X[base]);
        float4 r = *reinterpret_cast<const device float4*>(&R[base]);
        *reinterpret_cast<device float4*>(&Y[base]) = x + a * r;
    }
}
