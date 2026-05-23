// matrix_scale: B = alpha * A. Float4 grid-stride, memory-bound.
#include <metal_stdlib>
using namespace metal;

kernel void matrix_scale_f32(
    device const float*  A     [[buffer(0)]],
    device       float*  B     [[buffer(1)]],
    constant     uint&   N     [[buffer(2)]],
    constant     float&  alpha [[buffer(3)]],
    uint  tid                 [[thread_position_in_grid]])
{
    const uint grid_size = 64 * 1024;
    const uint n4 = N / 4;
    for (uint i = tid; i < n4; i += grid_size) {
        float4 v = *(reinterpret_cast<const device float4*>(&A[i * 4]));
        *(reinterpret_cast<device float4*>(&B[i * 4])) = alpha * v;
    }
    for (uint i = n4 * 4 + tid; i < N; i += grid_size)
        B[i] = alpha * A[i];
}
