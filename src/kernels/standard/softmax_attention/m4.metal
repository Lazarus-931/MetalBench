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
        uint j   = tid >> 3;
        uint sub = tid & 7;
        uint d0  = sub * 8;
        device const float* kr = K + j * D;
        float acc = 0.0f;
        acc += qrow[d0+0]*kr[d0+0];
        acc += qrow[d0+1]*kr[d0+1];
        acc += qrow[d0+2]*kr[d0+2];
        acc += qrow[d0+3]*kr[d0+3];
        acc += qrow[d0+4]*kr[d0+4];
        acc += qrow[d0+5]*kr[d0+5];
        acc += qrow[d0+6]*kr[d0+6];
        acc += qrow[d0+7]*kr[d0+7];
        acc += simd_shuffle_xor(acc, 1);
        acc += simd_shuffle_xor(acc, 2);
        acc += simd_shuffle_xor(acc, 4);
        if (sub == 0) scores[j] = acc * inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sg = tid >> 5;
    uint lane = tid & 31;

    float v = (tid < S) ? scores[tid] : -INFINITY;
    float mx_v = simd_max(v);
    if (lane == 0) reduce[sg] = mx_v;
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
        uint d  = tid >> 4;      // 0..63
        uint jl = tid & 15;      // 0..15
        uint j0 = jl * 8;
        float acc = 0.0f;
        acc += scores[j0+0] * V[(j0+0)*D + d];
        acc += scores[j0+1] * V[(j0+1)*D + d];
        acc += scores[j0+2] * V[(j0+2)*D + d];
        acc += scores[j0+3] * V[(j0+3)*D + d];
        acc += scores[j0+4] * V[(j0+4)*D + d];
        acc += scores[j0+5] * V[(j0+5)*D + d];
        acc += scores[j0+6] * V[(j0+6)*D + d];
        acc += scores[j0+7] * V[(j0+7)*D + d];
        acc += simd_shuffle_xor(acc, 1);
        acc += simd_shuffle_xor(acc, 2);
        acc += simd_shuffle_xor(acc, 4);
        acc += simd_shuffle_xor(acc, 8);
        if (jl == 0) O[qr * D + d] = acc;
    }
}
