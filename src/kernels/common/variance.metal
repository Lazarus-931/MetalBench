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
    const uint TG = 1024;
    const uint tid = tid3.x;
    const uint row = tgid.y;
    device const float* row_ptr = x + row * C;

    float sum = 0.0f, sumsq = 0.0f;
    for (uint i = tid; i < C; i += TG) {
        float v = row_ptr[i];
        sum   += v;
        sumsq += v * v;
    }

    sum   = simd_sum(sum);
    sumsq = simd_sum(sumsq);

    threadgroup float tg_s[32], tg_q[32];
    uint sg = tid >> 5;
    if ((tid & 31) == 0) { tg_s[sg] = sum; tg_q[sg] = sumsq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        sum   = tg_s[tid];
        sumsq = tg_q[tid];
        for (uint s = 16; s > 0; s >>= 1) {
            sum   += simd_shuffle_down(sum,   s);
            sumsq += simd_shuffle_down(sumsq, s);
        }
        if (tid == 0) {
            float mean = sum / float(C);
            out[row] = sumsq / float(C) - mean * mean;
        }
    }
}
