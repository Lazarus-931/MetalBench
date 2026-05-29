// softsign: out = x / (1 + |x|). float4 grid-stride with prefetch.
#include <metal_stdlib>
using namespace metal;

kernel void softsign_f32(
    device const float*  x         [[buffer(0)]],
    device       float*  y         [[buffer(1)]],
    constant     uint&   N         [[buffer(2)]],
    constant     uint&   grid_size [[buffer(3)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const uint n4 = N / 4;
    // Prefetch first chunk to reduce load latency
    float4 v0;
    if (tid < n4) {
        v0 = *reinterpret_cast<const device float4*>(&x[tid * 4]);
    }
    for (uint i = tid; i < n4; i += grid_size) {
        float4 v = v0;
        // Prefetch next chunk while computing current
        uint next_i = i + grid_size;
        if (next_i < n4) {
            v0 = *reinterpret_cast<const device float4*>(&x[next_i * 4]);
        }
        *reinterpret_cast<device float4*>(&y[i * 4]) = v / (1.0f + fabs(v));
    }
    for (uint i = n4 * 4 + tid; i < N; i += grid_size) {
        y[i] = x[i] / (1.0f + fabs(x[i]));
    }
}
