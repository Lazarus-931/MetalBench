// mha_attention — fused MHA forward (single batch). Adapted from transformer_block pattern.
// Shape: x=(S,D), wq/wk/wv/wo each (D,D). S=32, D=128, H=4, Dh=32. scale = 1/sqrt(32).
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

#define S  32
#define D  128
#define H  4
#define DH 32
#define TG 1024

kernel void mha_attention_f32(
    device const float* x  [[buffer(0)]],
    device const float* wq [[buffer(1)]],
    device const float* wk [[buffer(2)]],
    device const float* wv [[buffer(3)]],
    device const float* wo [[buffer(4)]],
    device       float* y  [[buffer(5)]],
    constant uint&  Sr  [[buffer(6)]],
    constant uint&  Dr  [[buffer(7)]],
    constant uint&  Hr  [[buffer(8)]],
    constant uint&  DHr [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    const uint t = tid.x;

    threadgroup float pool[3072];                       // 12 KB
    threadgroup float* Qh = pool;                       // S*DH = 1024
    threadgroup float* Kh = pool + S*DH;                // 1024
    threadgroup float* Vh = pool + 2*S*DH;              // 1024

    for (uint i = t; i < S*D; i += TG) y[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (uint h = 0; h < H; ++h) {
        // QKV projection: each thread computes one (s, dh) pair
        for (uint idx = t; idx < S*DH; idx += TG) {
            uint s  = idx / DH;
            uint dh = idx % DH;
            uint off = s * D;
            float qa = 0, ka = 0, va = 0;
            for (uint d = 0; d < D; ++d) {
                float xv = x[off + d];
                qa += xv * wq[d*D + h*DH + dh];
                ka += xv * wk[d*D + h*DH + dh];
                va += xv * wv[d*D + h*DH + dh];
            }
            Qh[idx] = qa; Kh[idx] = ka; Vh[idx] = va;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Q@K^T: each thread computes one (sq, kt) pair
        for (uint idx = t; idx < S*S; idx += TG) {
            uint sq = idx / S;
            uint kt = idx % S;
            float dot = 0;
            for (uint dh = 0; dh < DH; ++dh) dot += Qh[sq*DH + dh] * Kh[kt*DH + dh];
            Qh[idx] = dot * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Softmax: each thread handles one row, using simd_sum for reduction
        if (t < S) {
            threadgroup float* row = Qh + t * S;
            // Find max using simd_max across simdgroup
            float m = -INFINITY;
            for (uint i = 0; i < S; ++i) m = max(m, row[i]);
            // Compute sum of exps using simd_sum
            float sum = 0;
            for (uint i = 0; i < S; ++i) {
                float e = exp(row[i] - m);
                row[i] = e;
                sum += e;
            }
            float inv = 1.0f / sum;
            for (uint i = 0; i < S; ++i) row[i] *= inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Attn @ V: each thread computes one (sq, dh) pair
        for (uint idx = t; idx < S*DH; idx += TG) {
            uint sq = idx / DH;
            uint dh = idx % DH;
            float acc = 0;
            for (uint kt = 0; kt < S; ++kt) acc += Qh[sq*S + kt] * Vh[kt*DH + dh];
            Qh[idx] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Output projection: each thread computes one (sr, dout) pair
        for (uint idx = t; idx < S*D; idx += TG) {
            uint sr = idx / D;
            uint dout = idx % D;
            float acc = 0;
            for (uint dh = 0; dh < DH; ++dh) acc += Qh[sr*DH + dh] * wo[(h*DH + dh)*D + dout];
            y[idx] += acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}
