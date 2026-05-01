// trace: sum of diagonal A[i][i]. Single reduction via simd_sum.
#include <metal_stdlib>
using namespace metal;

kernel void trace_f32(
    device const float*  A   [[buffer(0)]],
    device       float*  out [[buffer(1)]],
    constant     uint&   N   [[buffer(2)]],
    uint  tid               [[thread_position_in_threadgroup]])
{
    const uint tg_size = 1024;
    float sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size)
        sum += A[i * N + i];

    float sg_sum = simd_sum(sum);

    threadgroup float tg[32];
    uint sg = tid >> 5;
    if ((tid & 31) == 0) tg[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        sg_sum = tg[tid];
        for (uint s = 16; s > 0; s >>= 1)
            sg_sum += simd_shuffle_down(sg_sum, s);
    }
    if (tid == 0) *out = sg_sum;
}
