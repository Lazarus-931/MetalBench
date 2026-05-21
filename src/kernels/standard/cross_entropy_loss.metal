// cross_entropy_loss: per-row -sum(y_onehot * log_softmax(logits)).
// Output shape (N,). One threadgroup per row.
#include <metal_stdlib>
using namespace metal;

kernel void cross_entropy_loss_f32(
    device const float*  L        [[buffer(0)]],   // logits (N, C)
    device const float*  Yh       [[buffer(1)]],   // one-hot (N, C)
    device       float*  out      [[buffer(2)]],   // (N,)
    constant     uint&   N        [[buffer(3)]],
    constant     uint&   C        [[buffer(4)]],
    uint3 tid3                   [[thread_position_in_threadgroup]],
    uint3 tgid                   [[threadgroup_position_in_grid]])
{
    const uint TG = 1024;
    const uint tid = tid3.x;
    const uint row = tgid.y;
    device const float* lp = L  + row * C;
    device const float* yp = Yh + row * C;

    threadgroup float reduce[32];

    float mx = -INFINITY;
    for (uint i = tid; i < C; i += TG) mx = fmax(mx, lp[i]);
    mx = simd_max(mx);
    if ((tid & 31) == 0) reduce[tid >> 5] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        mx = reduce[tid];
        for (uint s = 16; s > 0; s >>= 1) mx = fmax(mx, simd_shuffle_down(mx, s));
        if (tid == 0) reduce[0] = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = reduce[0];

    float sum = 0.0f;
    for (uint i = tid; i < C; i += TG) sum += exp(lp[i] - row_max);
    sum = simd_sum(sum);
    if ((tid & 31) == 0) reduce[tid >> 5] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        sum = reduce[tid];
        for (uint s = 16; s > 0; s >>= 1) sum += simd_shuffle_down(sum, s);
        if (tid == 0) reduce[0] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lse = row_max + log(reduce[0]);

    float nll = 0.0f;
    for (uint i = tid; i < C; i += TG) nll += yp[i] * (lse - lp[i]);
    nll = simd_sum(nll);
    if ((tid & 31) == 0) reduce[tid >> 5] = nll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        nll = reduce[tid];
        for (uint s = 16; s > 0; s >>= 1) nll += simd_shuffle_down(nll, s);
        if (tid == 0) out[row] = nll;
    }
}
