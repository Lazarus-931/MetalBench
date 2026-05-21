// argmax: per-row argmax along last dim. One threadgroup per row.
// Input (R, C). Output index as float (harness expects f32 outputs).
#include <metal_stdlib>
using namespace metal;

kernel void argmax_f32(
    device const float*  x       [[buffer(0)]],
    device       float*  out     [[buffer(1)]],
    constant     uint&   C       [[buffer(2)]],
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint TG = 1024;
    const uint tid = tid3.x;
    const uint row = tgid.y;
    device const float* row_ptr = x + row * C;

    float best_v = -INFINITY;
    uint  best_i = 0;
    for (uint i = tid; i < C; i += TG) {
        float v = row_ptr[i];
        if (v > best_v) { best_v = v; best_i = i; }
    }

    threadgroup float tg_v[32];
    threadgroup uint  tg_i[32];

    // Simdgroup reduction: pick max-value lane, propagate its index.
    for (uint s = 16; s > 0; s >>= 1) {
        float ov = simd_shuffle_down(best_v, s);
        uint  oi = simd_shuffle_down(best_i, s);
        if (ov > best_v) { best_v = ov; best_i = oi; }
    }
    uint sg = tid >> 5;
    if ((tid & 31) == 0) { tg_v[sg] = best_v; tg_i[sg] = best_i; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        best_v = tg_v[tid];
        best_i = tg_i[tid];
        for (uint s = 16; s > 0; s >>= 1) {
            float ov = simd_shuffle_down(best_v, s);
            uint  oi = simd_shuffle_down(best_i, s);
            if (ov > best_v) { best_v = ov; best_i = oi; }
        }
        if (tid == 0) out[row] = float(best_i);
    }
}
