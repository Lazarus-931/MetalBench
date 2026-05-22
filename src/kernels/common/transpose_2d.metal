// transpose_2d: B = A^T. One TG per 64x64 tile, 256 threads/TG.
// Each thread handles a 4x4 sub-block: float4 load row, then float4 store col.
// SMEM padded by +1 to avoid bank conflicts on the transposed access.
#include <metal_stdlib>
using namespace metal;

#define TILE 64

[[max_total_threads_per_threadgroup(256)]]
kernel void transpose_2d_f32(
    device const float*  A       [[buffer(0)]],
    device       float*  B       [[buffer(1)]],
    constant     uint&   M       [[buffer(2)]],
    constant     uint&   N       [[buffer(3)]],
    uint  lid                   [[thread_index_in_threadgroup]],
    uint  tgid                  [[threadgroup_position_in_grid]])
{
    threadgroup float tile[TILE][TILE + 1];

    const uint num_tiles_x = N / TILE;     // col-tiles in A
    const uint tx = tgid % num_tiles_x;
    const uint ty = tgid / num_tiles_x;

    const uint lx4 = lid & 15;   // 0..15  -> col group (4 floats wide)
    const uint ly  = lid >> 4;   // 0..15  -> row group (4 rows tall)

    // Load: thread (lx4,ly) reads 4 rows x float4 from A[ty*64 + ly*4 + r, tx*64 + lx4*4]
    #pragma unroll
    for (uint r = 0; r < 4; ++r) {
        const uint a_row = ty * TILE + ly * 4 + r;
        const uint a_col = tx * TILE + lx4 * 4;
        float4 v = *reinterpret_cast<const device float4*>(A + a_row * N + a_col);
        tile[lx4 * 4 + 0][ly * 4 + r] = v.x;
        tile[lx4 * 4 + 1][ly * 4 + r] = v.y;
        tile[lx4 * 4 + 2][ly * 4 + r] = v.z;
        tile[lx4 * 4 + 3][ly * 4 + r] = v.w;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Store: B[tx*64 + ly*4 + r, ty*64 + lx4*4] = tile[ly*4 + r, lx4*4 : +4]
    #pragma unroll
    for (uint r = 0; r < 4; ++r) {
        const uint b_row = tx * TILE + ly * 4 + r;
        const uint b_col = ty * TILE + lx4 * 4;
        float4 w = *reinterpret_cast<threadgroup float4*>(&tile[ly * 4 + r][lx4 * 4]);
        *reinterpret_cast<device float4*>(B + b_row * M + b_col) = w;
    }
}
