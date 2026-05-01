// transpose_2d: B = A^T. Coalesced reads, strided writes.
#include <metal_stdlib>
using namespace metal;

kernel void transpose_2d_f32(
    device const float*  A       [[buffer(0)]],
    device       float*  B       [[buffer(1)]],
    constant     uint&   M       [[buffer(2)]],
    constant     uint&   N       [[buffer(3)]],
    uint  tid                   [[thread_position_in_grid]])
{
    const uint grid_size = 64 * 1024; // hardcoded to match dispatch
    const uint total = M * N;
    for (uint idx = tid; idx < total; idx += grid_size) {
        uint i = idx / N;
        uint j = idx % N;
        B[j * M + i] = A[idx];
    }
}
