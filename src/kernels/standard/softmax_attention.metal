// softmax_attention: softmax(Q @ K^T / sqrt(D)) @ V. S=128, D=64.
// One TG per query row. TG=1024.
#include <metal_stdlib>
using namespace metal;

kernel void softmax_attention_f32(
    device const float*  Q       [[buffer(0)]],
    device const float*  K       [[buffer(1)]],
    device const float*  V       [[buffer(2)]],
    device       float*  O       [[buffer(3)]],
    constant     uint&   S       [[buffer(4)]],
    constant     uint&   D       [[buffer(5)]],
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint TG = 1024;
    const uint tid = tid3.x;
    const uint qr = tgid.y;
    if (qr >= S) return;

    threadgroup float qrow[64];
    threadgroup float scores[128];
    threadgroup float reduce[32];
    threadgroup float tmp[64 * 16];

    for (uint d = tid; d < D; d += TG) qrow[d] = Q[qr * D + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv = rsqrt(float(D));
    if (tid < S) {
        device const float* kr = K + tid * D;
        float dot = 0.0f;
        for (uint d = 0; d < D; ++d) dot += qrow[d] * kr[d];
        scores[tid] = dot * inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float v = (tid < S) ? scores[tid] : -INFINITY;
    float mx = simd_max(v);
    if ((tid & 31) == 0) reduce[tid >> 5] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float m = (tid < 4) ? reduce[tid] : -INFINITY;
        m = simd_max(m);
        if (tid == 0) reduce[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = reduce[0];

    float e = (tid < S) ? fast::exp(scores[tid] - row_max) : 0.0f;
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

    // Normalize scores in-place.
    if (tid < S) scores[tid] = scores[tid] * inv_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Output: O[qr, d] = sum_j scores[j] * V[j, d]. D=64, first 64 threads.
    // Output: O[qr, d] = sum_j scores[j] * V[j, d]. D=64, S=128.
    // Use all 1024 threads: tid = j_lane * D + d, with j_lane in [0, 16), d in [0, 64).
    // Each (j_lane, d) accumulates S/16 = 8 entries, then we reduce across j_lane.
    if (tid < 1024) {
        uint d = tid & 63;
        uint j_lane = tid >> 6;          // 0..15
        const uint J_GROUPS = 16;
        const uint per = S / J_GROUPS;   // 8
        float acc = 0.0f;
        uint j_start = j_lane * per;
        for (uint j = j_start; j < j_start + per; ++j) acc += scores[j] * V[j * D + d];
        // Reduce across j_lane (16 partials per d).
        tmp[d * 16 + j_lane] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (j_lane == 0) {
            float s = 0.0f;
            for (uint k = 0; k < 16; ++k) s += tmp[d * 16 + k];
            O[qr * D + d] = s;
        }
    }
}
