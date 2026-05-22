// matrix_add: C = A + B. Float4 grid-stride, 4-way unrolled for ILP.
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
    const device float4* A4 = reinterpret_cast<const device float4*>(A);
    const device float4* B4 = reinterpret_cast<const device float4*>(B);
    device       float4* C4 = reinterpret_cast<device float4*>(C);
    const uint n4 = N >> 2;
    const uint gs = grid_size;

    // With N=1M, n4=256K, gs=64K -> 4 iters per thread. Issue loads then stores.
    uint i0 = tid;
    uint i1 = i0 + gs;
    uint i2 = i1 + gs;
    uint i3 = i2 + gs;
    if (i3 < n4) {
        float4 a0 = A4[i0]; float4 b0 = B4[i0];
        float4 a1 = A4[i1]; float4 b1 = B4[i1];
        float4 a2 = A4[i2]; float4 b2 = B4[i2];
        float4 a3 = A4[i3]; float4 b3 = B4[i3];
        C4[i0] = a0 + b0;
        C4[i1] = a1 + b1;
        C4[i2] = a2 + b2;
        C4[i3] = a3 + b3;
    } else {
        for (uint i = tid; i < n4; i += gs) {
            C4[i] = A4[i] + B4[i];
        }
    }
    // tail
    for (uint i = n4 * 4 + tid; i < N; i += gs)
        C[i] = A[i] + B[i];
}
