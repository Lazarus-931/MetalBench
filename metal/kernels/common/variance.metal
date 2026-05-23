// variance: per-row Var(x) = mean(x²) − mean(x)². One threadgroup per row.
#include <metal_stdlib>
using namespace metal;

kernel void variance_f32(
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

    float v = row_ptr[tid];
    float sum   = simd_sum(v);
    float sumsq = simd_sum(v * v);

    threadgroup float tg_s[32], tg_q[32];
    if (lane == 0) { tg_s[sg] = sum; tg_q[sg] = sumsq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float s  = tg_s[lane];
        float sq = tg_q[lane];
        s  = simd_sum(s);
        sq = simd_sum(sq);
        if (lane == 0) {
            float mean = s / float(C);
            out[row] = sq / float(C) - mean * mean;
        }
    }
}
