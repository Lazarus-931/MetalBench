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
    const uint N4 = N / 4;
    device const float4* pred4 = reinterpret_cast<device const float4*>(pred);
    device const float4* targ4 = reinterpret_cast<device const float4*>(target);

    float4 acc = 0.0f;
    uint i = tid;
    for (; i + 3 * tg_size < N4; i += 4 * tg_size) {
        float4 da = pred4[i] - targ4[i];
        float4 db = pred4[i + tg_size] - targ4[i + tg_size];
        float4 dc = pred4[i + 2 * tg_size] - targ4[i + 2 * tg_size];
        float4 dd = pred4[i + 3 * tg_size] - targ4[i + 3 * tg_size];
        acc = fma(da, da, acc);
        acc = fma(db, db, acc);
        acc = fma(dc, dc, acc);
        acc = fma(dd, dd, acc);
    }
    for (; i < N4; i += tg_size) {
        float4 d = pred4[i] - targ4[i];
        acc = fma(d, d, acc);
    }
    float sum = acc.x + acc.y + acc.z + acc.w;

    float sg_sum = simd_sum(sum);

    threadgroup float tg_sum[32];
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    if (lane == 0) tg_sum[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float v = tg_sum[lane];
        v = simd_sum(v);
        if (lane == 0) *out = v / float(N);
    }
}
