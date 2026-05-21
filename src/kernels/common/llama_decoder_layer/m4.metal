// llama_decoder_layer — RMSNorm + GQA(+RoPE) + residual + RMSNorm + SwiGLU FFN + residual.
// S=64 D=128 H=4 HKV=2 Dh=32 FF=256.
#include <metal_stdlib>
using namespace metal;

#define S 64
#define D 128
#define H 4
#define HKV 2
#define DH 32
#define G 2
#define FF 256
#define TG 1024
#define QKV_W (D + 2*HKV*DH)

kernel void llama_decoder_layer_f32(
    device const float* x       [[buffer(0)]],
    device const float* W_qkv   [[buffer(1)]],
    device const float* W_o     [[buffer(2)]],
    device const float* W_gu    [[buffer(3)]],
    device const float* W_down  [[buffer(4)]],
    device       float* y       [[buffer(5)]],
    constant uint& S_  [[buffer(6)]],
    constant uint& D_  [[buffer(7)]],
    constant uint& H_  [[buffer(8)]],
    constant uint& HKV_ [[buffer(9)]],
    constant uint& FF_ [[buffer(10)]],
    constant float& base [[buffer(11)]],
    constant float& eps  [[buffer(12)]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    const uint t = tid.x;

    threadgroup float K[S*HKV*DH];
    threadgroup float V[S*HKV*DH];

    for (uint idx = t; idx < S*HKV*DH; idx += TG) {
        uint kv = idx / (S*DH);
        uint rem = idx % (S*DH);
        uint s = rem / DH;
        uint dh = rem % DH;
        uint i = dh / 2;
        uint pair_off = dh & 1;
        uint pair_dh0 = i*2;
        uint pair_dh1 = i*2+1;
        uint colK0 = D + kv*DH + pair_dh0;
        uint colK1 = D + kv*DH + pair_dh1;
        float ss = 0.0f;
        uint xoff = s * D;
        for (uint d = 0; d < D; ++d) { float v = x[xoff + d]; ss += v*v; }
        float rstd = rsqrt(ss / float(D) + eps);
        float k0v = 0.0f, k1v = 0.0f;
        for (uint d = 0; d < D; ++d) {
            float h_ = x[xoff + d] * rstd;
            k0v += h_ * W_qkv[d * QKV_W + colK0];
            k1v += h_ * W_qkv[d * QKV_W + colK1];
        }
        float omega = 1.0f / pow(base, (2.0f * float(i)) / float(DH));
        float ang = float(s) * omega;
        float cv = cos(ang);
        float sv = sin(ang);
        float rv;
        if (pair_off == 0) rv = k0v * cv - k1v * sv;
        else               rv = k0v * sv + k1v * cv;
        K[idx] = rv;
    }
    for (uint idx = t; idx < S*HKV*DH; idx += TG) {
        uint kv = idx / (S*DH);
        uint rem = idx % (S*DH);
        uint s = rem / DH;
        uint dh = rem % DH;
        uint colV = D + HKV*DH + kv*DH + dh;
        float ss = 0.0f;
        uint xoff = s * D;
        for (uint d = 0; d < D; ++d) { float v = x[xoff + d]; ss += v*v; }
        float rstd = rsqrt(ss / float(D) + eps);
        float vv = 0.0f;
        for (uint d = 0; d < D; ++d) vv += (x[xoff + d] * rstd) * W_qkv[d * QKV_W + colV];
        V[idx] = vv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float y_acc[8];
    for (uint i = 0; i < 8; ++i) {
        uint cell = t*8 + i;
        y_acc[i] = x[cell];
    }

    const float inv_sqrt_dh = rsqrt(float(DH));

    for (uint h_ = 0; h_ < H; ++h_) {
        const uint kv = h_ / G;

        for (uint idx = t; idx < S*DH; idx += TG) {
            uint s = idx / DH;
            uint dh = idx % DH;
            uint i = dh / 2;
            uint pair_off = dh & 1;
            uint pair_dh0 = i*2;
            uint pair_dh1 = i*2+1;
            uint colQ0 = h_*DH + pair_dh0;
            uint colQ1 = h_*DH + pair_dh1;
            float ss = 0.0f;
            uint xoff = s * D;
            for (uint d = 0; d < D; ++d) { float v = x[xoff + d]; ss += v*v; }
            float rstd = rsqrt(ss / float(D) + eps);
            float q0v = 0.0f, q1v = 0.0f;
            for (uint d = 0; d < D; ++d) {
                float h_v = x[xoff + d] * rstd;
                q0v += h_v * W_qkv[d * QKV_W + colQ0];
                q1v += h_v * W_qkv[d * QKV_W + colQ1];
            }
            float omega = 1.0f / pow(base, (2.0f * float(i)) / float(DH));
            float ang = float(s) * omega;
            float cv = cos(ang);
            float sv = sin(ang);
            float rv;
            if (pair_off == 0) rv = q0v * cv - q1v * sv;
            else               rv = q0v * sv + q1v * cv;
            y[idx] = rv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        {
            float my_sc[4];
            for (uint pi = 0; pi < 4; ++pi) {
                uint pair = t*4 + pi;
                uint sq = pair / S;
                uint kt = pair % S;
                float dot = 0;
                for (uint d = 0; d < DH; ++d) dot += y[sq*DH + d] * K[kv*S*DH + kt*DH + d];
                my_sc[pi] = dot * inv_sqrt_dh;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
            for (uint pi = 0; pi < 4; ++pi) y[2048u + t*4 + pi] = my_sc[pi];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        {
            uint sq = t >> 4;
            uint lane = t & 15;
            uint base_off = 2048u + sq * S;
            float v0 = y[base_off + lane];
            float v1 = y[base_off + lane + 16];
            float v2 = y[base_off + lane + 32];
            float v3 = y[base_off + lane + 48];
            float m = max(max(v0,v1), max(v2,v3));
            m = max(m, simd_shuffle_xor(m, 1));
            m = max(m, simd_shuffle_xor(m, 2));
            m = max(m, simd_shuffle_xor(m, 4));
            m = max(m, simd_shuffle_xor(m, 8));
            v0 = precise::exp(v0-m); v1 = precise::exp(v1-m); v2 = precise::exp(v2-m); v3 = precise::exp(v3-m);
            float ss = v0+v1+v2+v3;
            ss += simd_shuffle_xor(ss, 1);
            ss += simd_shuffle_xor(ss, 2);
            ss += simd_shuffle_xor(ss, 4);
            ss += simd_shuffle_xor(ss, 8);
            float inv = 1.0f / ss;
            y[base_off + lane]    = v0 * inv;
            y[base_off + lane+16] = v1 * inv;
            y[base_off + lane+32] = v2 * inv;
            y[base_off + lane+48] = v3 * inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        for (uint idx = t; idx < S*DH; idx += TG) {
            uint sq = idx / DH;
            uint dh = idx % DH;
            float acc = 0;
            for (uint k = 0; k < S; ++k) acc += y[2048u + sq*S + k] * V[kv*S*DH + k*DH + dh];
            y[6144u + idx] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        for (uint i = 0; i < 8; ++i) {
            uint cell = t*8 + i;
            uint s = cell / D;
            uint dout = cell % D;
            float acc = 0;
            for (uint dh = 0; dh < DH; ++dh) {
                acc += y[6144u + s*DH + dh] * W_o[(h_*DH + dh)*D + dout];
            }
            y_acc[i] += acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }

    for (uint i = 0; i < 8; ++i) y[t*8 + i] = y_acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    threadgroup float* ln_snap = K;
    {
        const uint row = t >> 4;
        const uint lane = t & 15;
        const uint off = row * D;
        float v[8];
        float ss = 0;
        for (uint i = 0; i < 8; ++i) {
            v[i] = y[off + lane + i*16];
            ss += v[i]*v[i];
        }
        ss += simd_shuffle_xor(ss, 1);
        ss += simd_shuffle_xor(ss, 2);
        ss += simd_shuffle_xor(ss, 4);
        ss += simd_shuffle_xor(ss, 8);
        float rstd_y = rsqrt(ss / float(D) + eps);
        for (uint i = 0; i < 8; ++i) ln_snap[off + lane + i*16] = v[i] * rstd_y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint sg = t >> 5;
    const uint lane = t & 31;

    for (uint pass = 0; pass < 2; ++pass) {
        const uint s = sg + pass * 32;
        const uint roff = s * D;
        const uint d0 = lane * 4;
        float l0 = ln_snap[roff + d0 + 0];
        float l1 = ln_snap[roff + d0 + 1];
        float l2 = ln_snap[roff + d0 + 2];
        float l3 = ln_snap[roff + d0 + 3];

        float o0 = 0, o1 = 0, o2 = 0, o3 = 0;

        for (uint f = 0; f < FF; ++f) {
            float wg0 = W_gu[(d0+0)*2u*FF + f];
            float wg1 = W_gu[(d0+1)*2u*FF + f];
            float wg2 = W_gu[(d0+2)*2u*FF + f];
            float wg3 = W_gu[(d0+3)*2u*FF + f];
            float pg = l0*wg0 + l1*wg1 + l2*wg2 + l3*wg3;
            float gate_f = simd_sum(pg);

            float wu0 = W_gu[(d0+0)*2u*FF + FF + f];
            float wu1 = W_gu[(d0+1)*2u*FF + FF + f];
            float wu2 = W_gu[(d0+2)*2u*FF + FF + f];
            float wu3 = W_gu[(d0+3)*2u*FF + FF + f];
            float pu = l0*wu0 + l1*wu1 + l2*wu2 + l3*wu3;
            float up_f = simd_sum(pu);

            float silu_g = gate_f / (1.0f + precise::exp(-gate_f));
            float h_f = silu_g * up_f;

            o0 = fma(h_f, W_down[f * D + d0 + 0], o0);
            o1 = fma(h_f, W_down[f * D + d0 + 1], o1);
            o2 = fma(h_f, W_down[f * D + d0 + 2], o2);
            o3 = fma(h_f, W_down[f * D + d0 + 3], o3);
        }

        y[roff + d0 + 0] += o0;
        y[roff + d0 + 1] += o1;
        y[roff + d0 + 2] += o2;
        y[roff + d0 + 3] += o3;
    }
}
