// attention_scores: softmax(Q @ K^T / sqrt(D)).
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

    threadgroup float qrow[64];      // D
    threadgroup float scores[128];   // S
    threadgroup float reduce[32];

    if (tid < D) qrow[tid] = Q[qr * D + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv = rsqrt(float(D));

    {
        uint j   = tid >> 3;
        uint sub = tid & 7;
        uint d0  = sub * 8;
        device const float* kr = K + j * D;
        float acc = 0.0f;
        acc += qrow[d0+0] * kr[d0+0];
        acc += qrow[d0+1] * kr[d0+1];
        acc += qrow[d0+2] * kr[d0+2];
        acc += qrow[d0+3] * kr[d0+3];
        acc += qrow[d0+4] * kr[d0+4];
        acc += qrow[d0+5] * kr[d0+5];
        acc += qrow[d0+6] * kr[d0+6];
        acc += qrow[d0+7] * kr[d0+7];
        acc += simd_shuffle_xor(acc, 1);
        acc += simd_shuffle_xor(acc, 2);
        acc += simd_shuffle_xor(acc, 4);
        if (sub == 0) scores[j] = acc * inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float v = (tid < S) ? scores[tid] : -INFINITY;
    float mx_v = simd_max(v);
    if ((tid & 31) == 0) reduce[tid >> 5] = mx_v;
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

    if (tid < S) O[qr * S + tid] = scores[tid] * inv_sum;
}
