// transpose_2d: B = A^T. Tiled via threadgroup memory for coalesced R/W.
#include <metal_stdlib>
using namespace metal;

#define TILE 32

kernel void transpose_2d_f32(
    device const float*  A       [[buffer(0)]],
    device       float*  B       [[buffer(1)]],
    constant     uint&   M       [[buffer(2)]],
    constant     uint&   N       [[buffer(3)]],
    uint  lid                   [[thread_index_in_threadgroup]],
    uint  tgid                  [[threadgroup_position_in_grid]])
{
    threadgroup float tile[TILE][TILE + 1];

    const uint num_tiles_x = N / TILE;          // along columns of A
    const uint num_tiles_y = M / TILE;          // along rows of A
    const uint total_tiles = num_tiles_x * num_tiles_y;
    const uint num_tgs = 64;                    // grid = 64*1024 / 1024

    const uint lx = lid & 31;                   // 0..31
    const uint ly = lid >> 5;                   // 0..31

    for (uint t = tgid; t < total_tiles; t += num_tgs) {
        const uint tx = t % num_tiles_x;
        const uint ty = t / num_tiles_x;
        const uint a_row = ty * TILE + ly;
        const uint a_col = tx * TILE + lx;
        tile[ly][lx] = A[a_row * N + a_col];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint b_row = tx * TILE + ly;
        const uint b_col = ty * TILE + lx;
        B[b_row * M + b_col] = tile[lx][ly];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
