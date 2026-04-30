// l1_norm: sum(abs(x)) per row. Simdgroup reduction.
#include <metal_stdlib>
using namespace metal;

kernel void l1_norm_f32(
    device const float*  x  [[buffer(0)]],
    device       float*  y  [[buffer(1)]],
    constant     uint&   N  [[buffer(2)]],
    uint3 tid              [[thread_position_in_threadgroup]],
    uint3 tgid             [[threadgroup_position_in_grid]])
{
    const uint t = tid.x;
    const uint row = tgid.y;
    float val = fabs(x[row * N + t]);
    float sum = simd_sum(val);

    threadgroup float tg_sum[32];
    const uint sg = t >> 5;
    if ((t & 31) == 0) tg_sum[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t < 32) {
        sum = tg_sum[t];
        for (uint s = 16; s > 0; s >>= 1)
            sum += simd_shuffle_down(sum, s);
    }
    if (t == 0) y[row] = sum;
}
