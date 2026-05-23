// outer_product: C[i][j] = x[i] * y[j]. Float4, 8-row unroll.
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
    const uint RPI = 8;
    const uint M2 = M / RPI;
    const uint total = M2 * n4;
    for (uint idx = tid; idx < total; idx += grid_size) {
        uint i2 = idx / n4;
        uint j4 = idx - i2 * n4;
        uint i = i2 * RPI;
        float x0 = x[i + 0];
        float x1 = x[i + 1];
        float x2 = x[i + 2];
        float x3 = x[i + 3];
        float x4 = x[i + 4];
        float x5 = x[i + 5];
        float x6 = x[i + 6];
        float x7 = x[i + 7];
        float4 yv = *(reinterpret_cast<const device float4*>(&y[j4 * 4]));
        device float4* Cp = reinterpret_cast<device float4*>(&C[i * N + j4 * 4]);
        Cp[0*n4] = x0 * yv;
        Cp[1*n4] = x1 * yv;
        Cp[2*n4] = x2 * yv;
        Cp[3*n4] = x3 * yv;
        Cp[4*n4] = x4 * yv;
        Cp[5*n4] = x5 * yv;
        Cp[6*n4] = x6 * yv;
        Cp[7*n4] = x7 * yv;
    }
    uint done = M2 * RPI;
    if (done < M) {
        uint rem = M - done;
        for (uint idx = tid; idx < rem * n4; idx += grid_size) {
            uint i = done + idx / n4;
            uint j4 = idx - (idx / n4) * n4;
            float xi = x[i];
            float4 yv = *(reinterpret_cast<const device float4*>(&y[j4 * 4]));
            *(reinterpret_cast<device float4*>(&C[i * N + j4 * 4])) = xi * yv;
        }
    }
}
