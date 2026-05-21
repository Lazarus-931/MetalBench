// matvec: y = A @ x. One threadgroup per row, float4 dot + simd reduce.
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
    // N=1024, 1024 thr per TG. Each thread handles 1 element -> can do float4 with 256 threads...
    // But registry gives 1024 thr/tg. Single element per thread.
    float sum = A[row * N + t] * x[t];
    float sg_sum = simd_sum(sum);

    threadgroup float tg[32];
    const uint sg = t >> 5;
    const uint lane = t & 31;
    if (lane == 0) tg[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float v = tg[lane];
        v = simd_sum(v);
        if (lane == 0) y[row] = v;
    }
}
