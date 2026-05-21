// attention_scores: softmax(Q @ K^T / sqrt(D)).
// One TG per query row. S=128, D=64. TG = 1024 threads, but only first 128 do work.
#include <metal_stdlib>
using namespace metal;

kernel void attention_scores_f32(
    device const float*  Q       [[buffer(0)]],
    device const float*  K       [[buffer(1)]],
    device       float*  O       [[buffer(2)]],
    constant     uint&   S       [[buffer(3)]],
    constant     uint&   D       [[buffer(4)]],
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint TG = 1024;
    const uint tid = tid3.x;
    const uint qr = tgid.y;
    if (qr >= S) return;

    threadgroup float qrow[64];      // D up to 64
    threadgroup float scores[128];   // S up to 128
    threadgroup float reduce[32];

    // Cooperatively load Q row.
    for (uint d = tid; d < D; d += TG) qrow[d] = Q[qr * D + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv = rsqrt(float(D));
    // Each of first S threads computes one score.
    if (tid < S) {
        device const float* kr = K + tid * D;
        float dot = 0.0f;
        for (uint d = 0; d < D; ++d) dot += qrow[d] * kr[d];
        scores[tid] = dot * inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Row max over S=128 elements.
    float v = (tid < S) ? scores[tid] : -INFINITY;
    float mx = simd_max(v);
    if ((tid & 31) == 0) reduce[tid >> 5] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        // 128/32 = 4 valid lanes
        float m = (tid < 4) ? reduce[tid] : -INFINITY;
        m = simd_max(m);
        if (tid == 0) reduce[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = reduce[0];

    float e = (tid < S) ? exp(scores[tid] - row_max) : 0.0f;
    if (tid < S) scores[tid] = e;
    float sum = simd_sum(e);
    if ((tid & 31) == 0) reduce[tid >> 5] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float s = (tid < 4) ? reduce[tid] : 0.0f;
        s = simd_sum(s);
        if (tid == 0) reduce[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / reduce[0];

    if (tid < S) O[qr * S + tid] = scores[tid] * inv_sum;
}
