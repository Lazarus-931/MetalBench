// matvec: y = A @ x. One threadgroup per row, simd_sum reduction.
#include <metal_stdlib>
using namespace metal;

kernel void matvec_f32(
    device const float*  A  [[buffer(0)]],
    device const float*  x  [[buffer(1)]],
    device       float*  y  [[buffer(2)]],
    constant     uint&   N  [[buffer(3)]],
    uint3 tid              [[thread_position_in_threadgroup]],
    uint3 tgid             [[threadgroup_position_in_grid]])
{
    const uint t = tid.x;
    const uint row = tgid.y;
    float sum = A[row * N + t] * x[t];
    float sg_sum = simd_sum(sum);

    threadgroup float tg[32];
    uint sg = t >> 5;
    if ((t & 31) == 0) tg[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t < 32) {
        sg_sum = tg[t];
        for (uint s = 16; s > 0; s >>= 1)
            sg_sum += simd_shuffle_down(sg_sum, s);
    }
    if (t == 0) y[row] = sg_sum;
}
