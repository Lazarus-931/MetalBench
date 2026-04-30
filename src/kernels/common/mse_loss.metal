// mse_loss: mean((pred - target)^2). Single-tg simdgroup reduction.
#include <metal_stdlib>
using namespace metal;

kernel void mse_loss_f32(
    device const float*  pred    [[buffer(0)]],
    device const float*  target  [[buffer(1)]],
    device       float*  out     [[buffer(2)]],
    constant     uint&   N       [[buffer(3)]],
    uint  tid                   [[thread_position_in_threadgroup]])
{
    const uint tg_size = 1024;
    float sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float d = pred[i] - target[i];
        sum += d * d;
    }

    float sg_sum = simd_sum(sum);

    threadgroup float tg_sum[32];
    uint sg = tid >> 5;
    if ((tid & 31) == 0) tg_sum[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        sg_sum = tg_sum[tid];
        for (uint s = 16; s > 0; s >>= 1)
            sg_sum += simd_shuffle_down(sg_sum, s);
    }
    if (tid == 0) *out = sg_sum / float(N);
}
