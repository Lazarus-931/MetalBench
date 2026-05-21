// swiglu: silu(x @ Wg) * (x @ Wu). M=N=K=256.
// Grid 1D, 256 thr/TG. Each TG handles one row of 256 outputs.
#include <metal_stdlib>
using namespace metal;

kernel void swiglu_f32(
    device const float*  X       [[buffer(0)]],
    device const float*  Wg      [[buffer(1)]],
    device const float*  Wu      [[buffer(2)]],
    device       float*  Y       [[buffer(3)]],
    constant     uint&   M       [[buffer(4)]],
    constant     uint&   N       [[buffer(5)]],
    constant     uint&   K       [[buffer(6)]],
    uint  tid                   [[thread_position_in_grid]],
    uint  lid                   [[thread_position_in_threadgroup]],
    uint  tgid                  [[threadgroup_position_in_grid]])
{
    threadgroup float xrow[256];
    const uint m = tgid;
    if (m >= M) return;

    xrow[lid] = X[m * K + lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = lid;
    float g = 0.0f, u = 0.0f;
    #pragma unroll(8)
    for (uint k = 0; k < K; ++k) {
        float xk = xrow[k];
        g += xk * Wg[k * N + n];
        u += xk * Wu[k * N + n];
    }
    float silu_g = g / (1.0f + fast::exp(-g));
    Y[m * N + n] = silu_g * u;
}
