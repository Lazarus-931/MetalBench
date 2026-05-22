// layer_norm: y = (x - mean) * rsqrt(var + eps) per row.
// 1024 thr/tg. Simdgroup reduction (fast) + tg reduction (5 steps).
#include <metal_stdlib>
using namespace metal;

kernel void layer_norm_f32(
    device const float*  x       [[buffer(0)]],
    device       float*  y       [[buffer(1)]],
    constant     uint&   N       [[buffer(2)]],
    constant     float&  eps     [[buffer(3)]],
    uint3 tid                   [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint t = tid.x;
    const uint row = tgid.y;
    const uint off = row * N;

    float val = x[off + t];
    float s2  = val * val;

    // Simdgroup reduction: 32 threads → 1 partial sum.
    float sum   = simd_sum(val);
    float sumsq = simd_sum(s2);

    // Threadgroup reduction across simdgroups (32 partial sums left).
    // Each simdgroup's first thread (t % 32 == 0) writes its partial sum.
    threadgroup float tg_sum[32];
    threadgroup float tg_sumsq[32];
    const uint sg = t >> 5;
    if ((t & 31) == 0) {
        tg_sum[sg]   = sum;
        tg_sumsq[sg] = sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First 32 threads reduce across simdgroups.
    if (t < 32) {
        sum   = tg_sum[t];
        sumsq = tg_sumsq[t];
        for (uint s = 16; s > 0; s >>= 1) {
            sum   += simd_shuffle_down(sum,   s);
            sumsq += simd_shuffle_down(sumsq, s);
        }
    }

    // Broadcast results to all threads via threadgroup memory.
    if (t < 32) {
        tg_sum[t] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum   = tg_sum[0]; // actually sum is the combined sum
    // Wait, sum from simd_shuffle_down at t=0 is the total sum.
    // But only thread 0 has the correct total. We need to broadcast.

    // Simpler: broadcast via tg memory.
    if (t == 0) {
        tg_sum[0] = sum;
        tg_sumsq[0] = sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = tg_sum[0] / float(N);
    float var  = max(tg_sumsq[0] / float(N) - mean * mean, 0.0f);
    float inv_std = rsqrt(var + eps);

    y[off + t] = (val - mean) * inv_std;
}
