// argmax: per-row argmax along last dim. One threadgroup per row.
#include <metal_stdlib>
using namespace metal;

static inline void simd_argmax(thread float& v, thread uint& i) {
    for (uint s = 16; s > 0; s >>= 1) {
        float ov = simd_shuffle_down(v, s);
        uint  oi = simd_shuffle_down(i, s);
        if (ov > v) { v = ov; i = oi; }
    }
}

kernel void argmax_f32(
    device const float*  x       [[buffer(0)]],
    device       float*  out     [[buffer(1)]],
    constant     uint&   C       [[buffer(2)]],
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint tid = tid3.x;
    const uint row = tgid.y;
    device const float* row_ptr = x + row * C;
    const uint sg = tid >> 5;
    const uint lane = tid & 31;

    float best_v = row_ptr[tid];
    uint  best_i = tid;

    simd_argmax(best_v, best_i);

    threadgroup float tg_v[32];
    threadgroup uint  tg_i[32];
    if (lane == 0) { tg_v[sg] = best_v; tg_i[sg] = best_i; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        best_v = tg_v[lane];
        best_i = tg_i[lane];
        simd_argmax(best_v, best_i);
        if (lane == 0) out[row] = float(best_i);
    }
}
