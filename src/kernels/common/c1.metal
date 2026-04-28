// Common Set #1 — sqr_matmul: square matrix multiply, C = A @ B  (N×N · N×N).
//
// Naïve baseline: one thread per output element, no tiling, no shared memory.
// This is the floor every other matmul kernel beats. Optimized variants
// (tiled, simdgroup_matrix, etc.) live in their own slots.
//
// Manifest contract (set in python/kernels/common/c1.py):
//   buffer(0): A — N*N f32 input
//   buffer(1): B — N*N f32 input
//   buffer(2): C — N*N f32 output
//   buffer(3): N — u32 scalar (matrix dimension)
//   grid:        (N, N, 1)
//   threadgroup: (16, 16, 1)
#include "utils.metal"

kernel void sqr_matmul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device       float* C [[buffer(2)]],
    constant     uint&  N [[buffer(3)]],
    uint2 gid             [[thread_position_in_grid]])
{
    const uint row = gid.y;
    const uint col = gid.x;
    if (row >= N || col >= N) return;

    float acc = 0.0f;
    for (uint k = 0; k < N; ++k) {
        acc += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}
