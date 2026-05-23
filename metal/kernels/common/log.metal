#include <metal_stdlib>
using namespace metal;

kernel void log_f32(
    device const float*  x         [[buffer(0)]],
    device       float*  y         [[buffer(1)]],
    constant     uint&   N         [[buffer(2)]],
    constant     uint&   grid_size [[buffer(3)]],
    uint  tid                     [[thread_position_in_grid]])
{
    // float4 vectorized + fast::log; N=262144 divisible by 4.
    const uint N4 = N >> 2;
    device const float4* x4 = reinterpret_cast<device const float4*>(x);
    device       float4* y4 = reinterpret_cast<device       float4*>(y);
    for (uint i = tid; i < N4; i += grid_size) {
        float4 v = fabs(x4[i]) + float4(1e-30f);
        y4[i] = float4(fast::log(v.x), fast::log(v.y), fast::log(v.z), fast::log(v.w));
    }
}
