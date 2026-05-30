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
    const uint lane = t & 31;

    float sum = simd_sum(fabs(x[off + t] - y[off + t]));

    threadgroup float tg[32];
    if (lane == 0) tg[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float v = tg[lane];
        v = simd_sum(v);
        if (lane == 0) out[row] = v;
    }
}
