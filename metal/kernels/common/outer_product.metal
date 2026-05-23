// outer_product: C[i][j] = x[i] * y[j]. Float4, memory-bound.
#include <metal_stdlib>
using namespace metal;

kernel void outer_product_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  y       [[buffer(1)]],
    device       float*  C       [[buffer(2)]],
    constant     uint&   M       [[buffer(3)]],
    constant     uint&   N       [[buffer(4)]],
    uint  tid                   [[thread_position_in_grid]])
{
    const uint grid_size = 64 * 1024;
    const uint n4 = N / 4;
    const uint total4 = M * n4;
    for (uint idx4 = tid; idx4 < total4; idx4 += grid_size) {
        uint i = idx4 / n4;
        uint j4 = idx4 - i * n4;
        float xi = x[i];
        float4 yv = *(reinterpret_cast<const device float4*>(&y[j4 * 4]));
        *(reinterpret_cast<device float4*>(&C[i * N + j4 * 4])) = xi * yv;
    }
}
