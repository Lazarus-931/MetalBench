// softmax_attention: softmax(Q @ K^T / sqrt(D)) @ V. S=128, D=64.
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

    if (tid < D) qrow[tid] = Q[qr * D + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv = rsqrt(float(D));

    {
        uint j = tid >> 3;        // 0..127
        uint sub = tid & 7;       // 0..7
        uint d0 = sub * 8;
        float acc = 0.0f;
        device const float* kr = K + j * D;
        for (uint dd = 0; dd < 8; ++dd) acc += qrow[d0 + dd] * kr[d0 + dd];
        acc += simd_shuffle_xor(acc, 1);
        acc += simd_shuffle_xor(acc, 2);
        acc += simd_shuffle_xor(acc, 4);
        if (sub == 0) scores[j] = acc * inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float v = (tid < S) ? scores[tid] : -INFINITY;
    float mx = simd_max(v);
    uint sg = tid >> 5;
    uint lane = tid & 31;
    if (lane == 0) reduce[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float m = (lane < 4) ? reduce[lane] : -INFINITY;
        m = simd_max(m);
        if (lane == 0) reduce[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = reduce[0];

    float e = (tid < S) ? fast::exp(scores[tid] - row_max) : 0.0f;
    if (tid < S) scores[tid] = e;
    float sum = simd_sum(e);
    if (lane == 0) reduce[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float s = (lane < 4) ? reduce[lane] : 0.0f;
        s = simd_sum(s);
        if (lane == 0) reduce[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / reduce[0];
    if (tid < S) scores[tid] = scores[tid] * inv_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        uint d = tid & 63;
        uint j_lane = tid >> 6;   // 0..15
        uint j_start = j_lane * 8;
        float acc = 0.0f;
        for (uint j = j_start; j < j_start + 8; ++j) acc += scores[j] * V[j * D + d];
        threadgroup float tmp[64 * 16];
        tmp[d * 16 + j_lane] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (j_lane == 0) {
            float s = 0.0f;
            for (uint k = 0; k < 16; ++k) s += tmp[d * 16 + k];
            O[qr * D + d] = s;
        }
    }
}
