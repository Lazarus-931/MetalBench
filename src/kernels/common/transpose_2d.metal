// transpose_2d: B = A^T. 64x64 tile per TG, float4 in/out, transposed SMEM.
// Each thread processes 2 tiles per loop iter to amortize control overhead.
#include <metal_stdlib>
using namespace metal;

#define TILE 64

[[max_total_threads_per_threadgroup(1024)]]
kernel void transpose_2d_f32(
    device const float*  A       [[buffer(0)]],
    device       float*  B       [[buffer(1)]],
    constant     uint&   M       [[buffer(2)]],
    constant     uint&   N       [[buffer(3)]],
    uint  lid                   [[thread_index_in_threadgroup]],
    uint  tgid                  [[threadgroup_position_in_grid]],
    uint  num_tgs               [[threadgroups_per_grid]])
{
    threadgroup float tile[TILE][TILE + 1];

    const uint num_tiles_x = N / TILE;
    const uint num_tiles_y = M / TILE;
    const uint total_tiles = num_tiles_x * num_tiles_y;

    const uint lx4 = lid & 15;
    const uint ly  = lid >> 4;

    for (uint t = tgid; t < total_tiles; t += num_tgs) {
        const uint tx = t % num_tiles_x;
        const uint ty = t / num_tiles_x;

        const uint a_row = ty * TILE + ly;
        const uint a_col = tx * TILE + lx4 * 4;
        float4 v = *reinterpret_cast<const device float4*>(A + a_row * N + a_col);

        tile[lx4 * 4 + 0][ly] = v.x;
        tile[lx4 * 4 + 1][ly] = v.y;
        tile[lx4 * 4 + 2][ly] = v.z;
        tile[lx4 * 4 + 3][ly] = v.w;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint b_row = tx * TILE + ly;
        const uint b_col = ty * TILE + lx4 * 4;
        float4 w = *reinterpret_cast<threadgroup float4*>(&tile[ly][lx4 * 4]);
        *reinterpret_cast<device float4*>(B + b_row * M + b_col) = w;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
