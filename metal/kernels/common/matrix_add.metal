// matrix_add: C = A + B. Process 2x float4 per iteration for better ILP.
#include <metal_stdlib>
using namespace metal;

kernel void matrix_add_f32(
    device const float4* A         [[buffer(0)]],
    device const float4* B         [[buffer(1)]],
    device       float4* C         [[buffer(2)]],
    constant     uint&   N         [[buffer(3)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const uint grid_size = 64u * 1024u;
    const uint n4 = N >> 2;

    // Each thread processes 4 float4 stride=grid_size
    // n4=262144, grid=65536, 4 iters/thread.
    // Unroll: prefetch all 4 loads, then issue stores.
    uint i0 = tid;
    uint i1 = tid + grid_size;
    uint i2 = tid + 2u * grid_size;
    uint i3 = tid + 3u * grid_size;

    float4 a0 = A[i0];
    float4 b0 = B[i0];
    float4 a1 = A[i1];
    float4 b1 = B[i1];
    float4 a2 = A[i2];
    float4 b2 = B[i2];
    float4 a3 = A[i3];
    float4 b3 = B[i3];

    C[i0] = a0 + b0;
    C[i1] = a1 + b1;
    C[i2] = a2 + b2;
    C[i3] = a3 + b3;

    // Tail (in case n4 isn't multiple of 4*grid_size)
    for (uint i = i3 + grid_size; i < n4; i += grid_size) {
        C[i] = A[i] + B[i];
    }
}
