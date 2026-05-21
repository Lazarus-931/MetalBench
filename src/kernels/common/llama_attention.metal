// llama_attention: GQA + RoPE attention (no KV cache, no LN).
// S=64 D=128 H=4 H_kv=2 D_head=32 (G = H/H_kv = 2)
// One TG of 1024 threads, 16 threads per query row.
#include <metal_stdlib>
using namespace metal;

#define S    64
#define D    128
#define HH   4
#define HKV  2
#define DH   32
#define G    2
#define TG   1024

kernel void llama_attention_f32(
    device const float* x      [[buffer(0)]],
    device const float* W_qkv  [[buffer(1)]],
    device const float* W_o    [[buffer(2)]],
    device       float* y      [[buffer(3)]],
    constant uint& S_   [[buffer(4)]],
    constant uint& D_   [[buffer(5)]],
    constant uint& H_   [[buffer(6)]],
    constant uint& Hkv_ [[buffer(7)]],
    constant float& base [[buffer(8)]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    const uint t = tid.x;
    constexpr uint QKV_W = D + 2u * HKV * DH;   // 256

    threadgroup float Kh[S * DH];
    threadgroup float Vh[S * DH];
    threadgroup float oh[S * DH];

    const float inv_sqrt_dh = rsqrt(float(DH));
    const float two_over_dh = 2.0f / float(DH);

    for (uint kvh = 0; kvh < HKV; ++kvh) {
        // K, V projection (with RoPE on K).
        {
            const uint sr   = t >> 4;
            const uint lane = t & 15;
            const uint off  = sr * D;
            const uint i_pair = lane;
            const uint dh0 = 2 * i_pair;
            const uint dh1 = dh0 + 1;
            const uint colK0 = D + kvh*DH + dh0;
            const uint colK1 = D + kvh*DH + dh1;
            const uint colV0 = D + HKV*DH + kvh*DH + dh0;
            const uint colV1 = D + HKV*DH + kvh*DH + dh1;
            float k0 = 0.f, k1 = 0.f, v0 = 0.f, v1 = 0.f;
            for (uint d = 0; d < D; ++d) {
                float xv = x[off + d];
                k0 += xv * W_qkv[d*QKV_W + colK0];
                k1 += xv * W_qkv[d*QKV_W + colK1];
                v0 += xv * W_qkv[d*QKV_W + colV0];
                v1 += xv * W_qkv[d*QKV_W + colV1];
            }
            float pos = float(sr);
            float omega = precise::pow(base, -two_over_dh * float(i_pair));
            float ang = pos * omega;
            float c = precise::cos(ang), s_ = precise::sin(ang);
            Kh[sr*DH + dh0] = k0 * c - k1 * s_;
            Kh[sr*DH + dh1] = k0 * s_ + k1 * c;
            Vh[sr*DH + dh0] = v0;
            Vh[sr*DH + dh1] = v1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint h_off = 0; h_off < G; ++h_off) {
            const uint h = kvh * G + h_off;
            {
                const uint sq   = t >> 4;
                const uint lane = t & 15;
                const uint off  = sq * D;
                const uint i_pair = lane;
                const uint dh0 = 2 * i_pair;
                const uint dh1 = dh0 + 1;
                const uint colQ0 = h*DH + dh0;
                const uint colQ1 = h*DH + dh1;
                float q0 = 0.f, q1 = 0.f;
                for (uint d = 0; d < D; ++d) {
                    float xv = x[off + d];
                    q0 += xv * W_qkv[d*QKV_W + colQ0];
                    q1 += xv * W_qkv[d*QKV_W + colQ1];
                }
                float pos = float(sq);
                float omega = precise::pow(base, -two_over_dh * float(i_pair));
                float ang = pos * omega;
                float c = precise::cos(ang), s_ = precise::sin(ang);
                float q0r = q0 * c - q1 * s_;
                float q1r = q0 * s_ + q1 * c;

                float out0 = 0.f, out1 = 0.f;
                float m_run = -INFINITY;
                float l_run = 0.f;
                for (uint kt = 0; kt < S; ++kt) {
                    float ka = Kh[kt*DH + dh0];
                    float kb = Kh[kt*DH + dh1];
                    float partial = q0r * ka + q1r * kb;
                    partial += simd_shuffle_xor(partial, 1);
                    partial += simd_shuffle_xor(partial, 2);
                    partial += simd_shuffle_xor(partial, 4);
                    partial += simd_shuffle_xor(partial, 8);
                    float score = partial * inv_sqrt_dh;
                    float mn = max(m_run, score);
                    float alpha = exp(m_run - mn);
                    float beta  = exp(score - mn);
                    out0 = out0 * alpha + beta * Vh[kt*DH + dh0];
                    out1 = out1 * alpha + beta * Vh[kt*DH + dh1];
                    l_run = l_run * alpha + beta;
                    m_run = mn;
                }
                float inv = 1.0f / l_run;
                oh[sq*DH + dh0] = out0 * inv;
                oh[sq*DH + dh1] = out1 * inv;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint idx = t; idx < S*D; idx += TG) {
                const uint sr   = idx / D;
                const uint dout = idx % D;
                float acc = 0.f;
                for (uint dh = 0; dh < DH; ++dh) {
                    acc += oh[sr*DH + dh] * W_o[(h*DH + dh)*D + dout];
                }
                if (kvh == 0 && h_off == 0) {
                    y[sr*D + dout] = acc;
                } else {
                    y[sr*D + dout] += acc;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        }
    }
}
