// exp: float4 grid-stride loop. Memory-bound elementwise.
#include <metal_stdlib>
using namespace metal;

kernel void exp_f32(
    device const float*  x         [[buffer(0)]],
    device       float*  y         [[buffer(1)]],
    constant     uint&   N         [[buffer(2)]],
    constant     uint&   grid_size [[buffer(3)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const uint n4 = N / 4;
    for (uint i = tid; i < n4; i += grid_size) {
        float4 v = *(reinterpret_cast<const device float4*>(&x[i * 4]));
        float4 r;
        r.x = fast::exp(v.x);
        r.y = fast::exp(v.y);
        r.z = fast::exp(v.z);
        r.w = fast::exp(v.w);
        *(reinterpret_cast<device float4*>(&y[i * 4])) = r;
    }
    for (uint i = n4 * 4 + tid; i < N; i += grid_size) {
        y[i] = fast::exp(x[i]);
    }
}
