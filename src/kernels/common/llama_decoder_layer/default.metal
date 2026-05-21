// llama_decoder_layer: RMSNorm + GQA(+RoPE) + residual + RMSNorm + SwiGLU FFN + residual.
// S=64 D=128 H=4 H_kv=2 D_head=32 FF=256 (G = H/H_kv = 2). One TG of 1024 threads.
#include <metal_stdlib>
using namespace metal;

#define S    64
#define D    128
#define HH   4
#define HKV  2
#define DH   32
#define G    2
#define FF   256
#define TG   1024

inline float silu(float v) { return v / (1.0f + exp(-v)); }

kernel void llama_decoder_layer_f32(
    device const float* x      [[buffer(0)]],
    device const float* W_qkv  [[buffer(1)]],
    device const float* W_o    [[buffer(2)]],
    device const float* W_gu   [[buffer(3)]],
    device const float* W_down [[buffer(4)]],
    device       float* y      [[buffer(5)]],
    constant uint& S_   [[buffer(6)]],
    constant uint& D_   [[buffer(7)]],
    constant uint& H_   [[buffer(8)]],
    constant uint& Hkv_ [[buffer(9)]],
    constant uint& FF_  [[buffer(10)]],
    constant float& base [[buffer(11)]],
    constant float& eps  [[buffer(12)]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    const uint t = tid.x;
    constexpr uint QKV_W = D + 2u * HKV * DH;
    constexpr uint GU_W  = 2u * FF;

    threadgroup float rstd[S];                  // 64
    threadgroup float Kh[S * DH];               // 2048
    threadgroup float Vh[S * DH];               // 2048
    threadgroup float oh[S * DH];               // 2048
    // Total: 64 + 6144 = 6208 floats = ~24.5 KB
    // After attention: reuse Kh + Vh + oh as ff_out (6144 floats covers most of y; need 8192).
    // For FFN we accumulate into y itself by computing one row at a time (no y-read/write race).

    const float inv_sqrt_dh = rsqrt(float(DH));
    const float two_over_dh = 2.0f / float(DH);

    // RMSNorm rstd over x.
    {
        const uint sr   = t >> 4;
        const uint lane = t & 15;
        const uint off  = sr * D;
        float s2 = 0.f;
        for (uint d = lane; d < D; d += 16) {
            float v = x[off + d];
            s2 += v * v;
        }
        s2 += simd_shuffle_xor(s2, 1);
        s2 += simd_shuffle_xor(s2, 2);
        s2 += simd_shuffle_xor(s2, 4);
        s2 += simd_shuffle_xor(s2, 8);
        if (lane == 0) rstd[sr] = rsqrt(s2 / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // y = x.
    for (uint i = t; i < S*D; i += TG) y[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // GQA attention.
    for (uint kvh = 0; kvh < HKV; ++kvh) {
        // K, V projection + RoPE on K.
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
            const float rs = rstd[sr];
            float k0 = 0.f, k1 = 0.f, v0 = 0.f, v1 = 0.f;
            for (uint d = 0; d < D; ++d) {
                float ln = x[off + d] * rs;
                k0 += ln * W_qkv[d*QKV_W + colK0];
                k1 += ln * W_qkv[d*QKV_W + colK1];
                v0 += ln * W_qkv[d*QKV_W + colV0];
                v1 += ln * W_qkv[d*QKV_W + colV1];
            }
            float omega = precise::pow(base, -two_over_dh * float(i_pair));
            float ang = float(sr) * omega;
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
                const float rs = rstd[sq];
                float q0 = 0.f, q1 = 0.f;
                for (uint d = 0; d < D; ++d) {
                    float ln = x[off + d] * rs;
                    q0 += ln * W_qkv[d*QKV_W + colQ0];
                    q1 += ln * W_qkv[d*QKV_W + colQ1];
                }
                float omega = precise::pow(base, -two_over_dh * float(i_pair));
                float ang = float(sq) * omega;
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
                y[sr*D + dout] += acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        }
    }

    // RMSNorm rstd over y.
    {
        const uint sr   = t >> 4;
        const uint lane = t & 15;
        const uint off  = sr * D;
        float s2 = 0.f;
        for (uint d = lane; d < D; d += 16) {
            float v = y[off + d];
            s2 += v * v;
        }
        s2 += simd_shuffle_xor(s2, 1);
        s2 += simd_shuffle_xor(s2, 2);
        s2 += simd_shuffle_xor(s2, 4);
        s2 += simd_shuffle_xor(s2, 8);
        if (lane == 0) rstd[sr] = rsqrt(s2 / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Row-serial SwiGLU FFN — avoids the y-read/y-write race across tiles.
    // We have Kh+Vh+oh = 6144 floats available (attention done).
    // Per-row buffers:
    //   ln_row[D=128]      = LN(y)[sr, :]
    //   tile[FF=256]       = silu(gate)*up for this row
    // Total per row: 384 floats. Use the start of Kh.
    threadgroup float* ln_row = Kh;          // [D]
    threadgroup float* tile   = Kh + D;      // [FF]

    for (uint sr = 0; sr < S; ++sr) {
        const float rs = rstd[sr];
        const uint row_off = sr * D;

        // Load ln_row.
        for (uint d = t; d < D; d += TG) {
            ln_row[d] = y[row_off + d] * rs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // tile[f] = silu(gate)*up; one thread per f.
        for (uint f = t; f < FF; f += TG) {
            float gate_acc = 0.f, up_acc = 0.f;
            for (uint d = 0; d < D; ++d) {
                float ln = ln_row[d];
                gate_acc += ln * W_gu[d*GU_W + f];
                up_acc   += ln * W_gu[d*GU_W + FF + f];
            }
            tile[f] = silu(gate_acc) * up_acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // y[sr, d_out] += sum_f tile[f] * W_down[f, d_out]; one thread per d_out (D=128).
        for (uint d_out = t; d_out < D; d_out += TG) {
            float acc = 0.f;
            for (uint f = 0; f < FF; ++f) {
                acc += tile[f] * W_down[f*D + d_out];
            }
            y[row_off + d_out] += acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}
