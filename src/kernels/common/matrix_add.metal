// matrix_add: C = A + B. Float4 grid-stride, memory-bound.
#include <metal_stdlib>
using namespace metal;

kernel void matrix_add_f32(
    device const float*  A         [[buffer(0)]],
    device const float*  B         [[buffer(1)]],
    device       float*  C         [[buffer(2)]],
    constant     uint&   N         [[buffer(3)]],
    uint  tid                     [[thread_position_in_grid]],
    uint  grid_size               [[threads_per_grid]])
{
    const uint n4 = N / 4;
    for (uint i = tid; i < n4; i += grid_size) {
        float4 a = *(reinterpret_cast<const device float4*>(&A[i * 4]));
        float4 b = *(reinterpret_cast<const device float4*>(&B[i * 4]));
        *(reinterpret_cast<device float4*>(&C[i * 4])) = a + b;
    }
    for (uint i = n4 * 4 + tid; i < N; i += grid_size)
        C[i] = A[i] + B[i];
}
