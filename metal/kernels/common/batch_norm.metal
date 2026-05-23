// batchnorm: normalize each column over rows. (1024, 256) input.
// TG=256 handles 2 columns. 128 threads per column, each thread 8 rows.
#include <metal_stdlib>
using namespace metal;

kernel void batch_norm_f32(
    device const float*  x       [[buffer(0)]],
    device       float*  y       [[buffer(1)]],
    constant     uint&   N       [[buffer(2)]],
    constant     uint&   C       [[buffer(3)]],
    constant     float&  eps     [[buffer(4)]],
    uint3 tid                   [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint t = tid.x;
    // Each TG handles 2 cols: col_base = tgid.y * 2, sub = t / 128.
    const uint sub = t >> 7;             // 0 or 1
    const uint local = t & 127;          // 0..127
    const uint col = (tgid.y << 1) + sub;

    // 8 rows per thread, stride 128.
    float vv[8];
    float sum = 0.0f, sumsq = 0.0f;
    #pragma unroll
    for (uint k = 0; k < 8; ++k) {
        float v = x[(local + k * 128) * C + col];
        vv[k] = v;
        sum += v;
        sumsq += v * v;
    }

    sum   = simd_sum(sum);
    sumsq = simd_sum(sumsq);

    // Each col has 4 simdgroups. Two cols → 8 simdgroups total.
    // sg index in TG: 0..7. col0 → sg 0..3, col1 → sg 4..7.
    threadgroup float tg_sum[8];
    threadgroup float tg_sumsq[8];
    const uint sg = t >> 5;
    if ((t & 31) == 0) {
        tg_sum[sg]   = sum;
        tg_sumsq[sg] = sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First 8 threads each pick a slot; reduce within each 4-slot half.
    // Simpler: thread 0 of each half reduces 4 slots.
    if (local == 0) {
        uint base = sub * 4;
        float s = tg_sum[base] + tg_sum[base+1] + tg_sum[base+2] + tg_sum[base+3];
        float s2 = tg_sumsq[base] + tg_sumsq[base+1] + tg_sumsq[base+2] + tg_sumsq[base+3];
        tg_sum[base]   = s;
        tg_sumsq[base] = s2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum   = tg_sum[sub * 4];
    float total_sumsq = tg_sumsq[sub * 4];

    float mean = total_sum / float(N);
    float var  = max(total_sumsq / float(N) - mean * mean, 0.0f);
    float inv_std = rsqrt(var + eps);

    #pragma unroll
    for (uint k = 0; k < 8; ++k) {
        y[(local + k * 128) * C + col] = (vv[k] - mean) * inv_std;
    }
}
