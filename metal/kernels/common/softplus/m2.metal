// softplus: log(1 + exp(x)). Stable form: max(x,0) + log1p(exp(-|x|)).
// Memory-bound kernel: use vector loads/stores and minimize thread dispatch overhead.
#include <metal_stdlib>
using namespace metal;

kernel void softplus_f32(
    device const float*  x         [[buffer(0)]],
    device       float*  y         [[buffer(1)]],
    constant     uint&   N         [[buffer(2)]],
    constant     uint&   grid_size [[buffer(3)]],
    uint  tid                     [[thread_position_in_grid]])
{
    // Process 8 elements per thread to reduce grid size and dispatch overhead
    // while keeping memory access coalesced via float4 loads.
    const uint n8 = N / 8;
    for (uint i = tid; i < n8; i += grid_size) {
        float4 v0 = *reinterpret_cast<const device float4*>(&x[i * 8]);
        float4 v1 = *reinterpret_cast<const device float4*>(&x[i * 8 + 4]);
        
        float4 ax0 = fabs(v0);
        float4 mx0 = fmax(v0, 0.0f);
        float4 ax1 = fabs(v1);
        float4 mx1 = fmax(v1, 0.0f);
        
        *reinterpret_cast<device float4*>(&y[i * 8]) = mx0 + log(1.0f + exp(-ax0));
        *reinterpret_cast<device float4*>(&y[i * 8 + 4]) = mx1 + log(1.0f + exp(-ax1));
    }
    // Handle remaining elements (0-7) that don't fit in the 8-element chunks
    for (uint i = n8 * 8 + tid; i < N; i += grid_size) {
        float v = x[i];
        y[i] = fmax(v, 0.0f) + log(1.0f + exp(-fabs(v)));
    }
}
