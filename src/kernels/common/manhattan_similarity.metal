// manhattan_similarity: L1 distance per row. simd_sum(|x - y|).
#include <metal_stdlib>
using namespace metal;

kernel void manhattan_similarity_f32(
    device const float*  x   [[buffer(0)]],
    device const float*  y   [[buffer(1)]],
    device       float*  out [[buffer(2)]],
    constant     uint&   D   [[buffer(3)]],
    uint3 tid               [[thread_position_in_threadgroup]],
    uint3 tgid              [[threadgroup_position_in_grid]])
{
    const uint t = tid.x;
    const uint row = tgid.y;
    const uint off = row * D;
    const uint sg = t >> 5;

    float sum = simd_sum(fabs(x[off + t] - y[off + t]));

    threadgroup float tg[32];
    if ((t & 31) == 0) tg[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t < 32) {
        sum = tg[t];
        for (uint s = 16; s > 0; s >>= 1)
            sum += simd_shuffle_down(sum, s);
    }
    if (t == 0) out[row] = sum;
}
