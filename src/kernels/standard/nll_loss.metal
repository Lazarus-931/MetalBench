// nll_loss: per-row -sum(y_onehot * log_probs). Output (N,).
#include <metal_stdlib>
using namespace metal;

kernel void nll_loss_f32(
    device const float*  LP       [[buffer(0)]],
    device const float*  Yh       [[buffer(1)]],
    device       float*  out      [[buffer(2)]],
    constant     uint&   N        [[buffer(3)]],
    constant     uint&   C        [[buffer(4)]],
    uint3 tid3                   [[thread_position_in_threadgroup]],
    uint3 tgid                   [[threadgroup_position_in_grid]])
{
    const uint TG = 1024;
    const uint tid = tid3.x;
    const uint row = tgid.y;
    device const float* lp = LP + row * C;
    device const float* yp = Yh + row * C;
    threadgroup float reduce[32];

    // C = 1024, TG = 1024 → each thread handles 1 element. Use float4 instead.
    // But TG=1024 and C=1024 means tid covers all. Use single load.
    float nll = -yp[tid] * lp[tid];
    nll = simd_sum(nll);
    if ((tid & 31) == 0) reduce[tid >> 5] = nll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        nll = simd_sum(reduce[tid]);
        if (tid == 0) out[row] = nll;
    }
}
