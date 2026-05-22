// where: y = cond > 0.5 ? a : b. Float4 grid-stride.
#include <metal_stdlib>
using namespace metal;

kernel void where_f32(
    device const float*  cond      [[buffer(0)]],
    device const float*  a         [[buffer(1)]],
    device const float*  b         [[buffer(2)]],
    device       float*  y         [[buffer(3)]],
    constant     uint&   N         [[buffer(4)]],
    constant     uint&   grid_size [[buffer(5)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const device float4* cond4 = reinterpret_cast<const device float4*>(cond);
    const device float4* a4    = reinterpret_cast<const device float4*>(a);
    const device float4* b4    = reinterpret_cast<const device float4*>(b);
    device       float4* y4    = reinterpret_cast<device float4*>(y);

    const uint n4 = N >> 2;
    const uint gs = grid_size;
    for (uint i = tid; i < n4; i += gs) {
        float4 c = cond4[i];
        float4 av = a4[i];
        float4 bv = b4[i];
        y4[i] = select(bv, av, c > 0.5f);
    }
    for (uint i = n4 * 4 + tid; i < N; i += gs) {
        y[i] = cond[i] > 0.5f ? a[i] : b[i];
    }
}
