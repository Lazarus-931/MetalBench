// linear_bias: y = x @ W + b. (M,K) @ (K,N) + (N,). M=N=K=256.
// Grid: 1D, 256 thr/tg. Each TG processes 256 contiguous output elements
// — exactly one full row of N=256 outputs.
#include <metal_stdlib>
using namespace metal;

kernel void linear_bias_f32(
    device const float*  X        [[buffer(0)]],
    device const float*  W        [[buffer(1)]],
    device const float*  B        [[buffer(2)]],
    device       float*  Y        [[buffer(3)]],
    constant     uint&   M        [[buffer(4)]],
    constant     uint&   N        [[buffer(5)]],
    constant     uint&   K        [[buffer(6)]],
    uint tid                     [[thread_position_in_grid]],
    uint lid                     [[thread_position_in_threadgroup]],
    uint tgid                    [[threadgroup_position_in_grid]])
{
    threadgroup float xrow[256];     // K = 256
    threadgroup float brow[256];     // N = 256
    const uint m = tgid;             // row index
    if (m >= M) return;

    // Cooperatively load X row (K=256) and bias (N=256). Both 256 floats with 256 threads.
    xrow[lid] = X[m * K + lid];
    brow[lid] = B[lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread computes one output col n = lid.
    const uint n = lid;
    float acc = brow[n];
    #pragma unroll(8)
    for (uint k = 0; k < K; ++k) acc += xrow[k] * W[k * N + n];
    Y[m * N + n] = acc;
}
