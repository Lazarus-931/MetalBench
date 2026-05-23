// variance: per-row Var(x) = mean(x²) − mean(x)². One threadgroup per row.
// TG=256, each thread does a float4 load (covers C=1024).
#include <metal_stdlib>
using namespace metal;

kernel void variance_f32(
    device const float*  x       [[buffer(0)]],
    device       float*  out     [[buffer(1)]],
    constant     uint&   C       [[buffer(2)]],
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint tid  = tid3.x;
    const uint row  = tgid.y;
    const uint sg   = tid >> 5;
    const uint lane = tid & 31;

    device const float4* row4 = (device const float4*)(x + row * C);
    float4 v = row4[tid];
    float s_local = v.x + v.y + v.z + v.w;
    float q_local = v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;

    float sum   = simd_sum(s_local);
    float sumsq = simd_sum(q_local);

    threadgroup float tg_s[8], tg_q[8];
    if (lane == 0) { tg_s[sg] = sum; tg_q[sg] = sumsq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0 && lane < 8) {
        float s  = tg_s[lane];
        float sq = tg_q[lane];
        s  += simd_shuffle_xor(s, 4);
        s  += simd_shuffle_xor(s, 2);
        s  += simd_shuffle_xor(s, 1);
        sq += simd_shuffle_xor(sq, 4);
        sq += simd_shuffle_xor(sq, 2);
        sq += simd_shuffle_xor(sq, 1);
        if (lane == 0) {
            float invC = 1.0f / float(C);
            float mean = s * invC;
            out[row] = sq * invC - mean * mean;
        }
    }
}
