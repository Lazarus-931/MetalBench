// transformer_block — pre-LN BERT/ViT block, M2 redesign
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define S 64
#define D 128
#define H 4
#define DH 32
#define FF 256
#define FF_BLK 64
#define NCHUNKS (FF / FF_BLK)   // 4
#define TG 1024

kernel void transformer_block_f32(
    device const float* x      [[buffer(0)]],
    device const float* W_qkv  [[buffer(1)]],
    device const float* W_o    [[buffer(2)]],
    device const float* W_ff1  [[buffer(3)]],
    device const float* W_ff2  [[buffer(4)]],
    device       float* y      [[buffer(5)]],
    constant uint& S_  [[buffer(6)]],
    constant uint& D_  [[buffer(7)]],
    constant uint& H_  [[buffer(8)]],
    constant uint& FF_ [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]])
{
    const uint t = tid.x;
    threadgroup float pool[8192];                  // 32 KB

    threadgroup float* means = pool;            // [64]
    threadgroup float* rstds = pool + 64;       // [64]
    threadgroup float* Qh = pool + 128;         // [S*DH=2048]
    threadgroup float* Kh = pool + 128 + 2048;
    threadgroup float* Vh = pool + 128 + 4096;
    threadgroup float* sc = pool + 128;

    {
        const uint row = t >> 4;
        const uint lane = t & 15;
        const uint off = row * D;
        float s1 = 0, s2 = 0;
        for (uint d = lane; d < D; d += 16) {
            float v = x[off + d]; s1 += v; s2 += v*v;
        }
        s1 += simd_shuffle_xor(s1, 1);
        s1 += simd_shuffle_xor(s1, 2);
        s1 += simd_shuffle_xor(s1, 4);
        s1 += simd_shuffle_xor(s1, 8);
        s2 += simd_shuffle_xor(s2, 1);
        s2 += simd_shuffle_xor(s2, 2);
        s2 += simd_shuffle_xor(s2, 4);
        s2 += simd_shuffle_xor(s2, 8);
        if (lane == 0) {
            float m = s1 / float(D);
            float v = max(s2/float(D) - m*m, 0.0f);
            means[row] = m;
            rstds[row] = rsqrt(v + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_sqrt_dh = rsqrt(float(DH));
    for (uint h = 0; h < H; ++h) {
        for (uint idx = t; idx < S*DH; idx += TG) {
            uint s = idx / DH;
            uint dh = idx % DH;
            uint off = s * D;
            float m = means[s], rs = rstds[s];
            float qa = 0, ka = 0, va = 0;
            for (uint d = 0; d < D; ++d) {
                float ln = (x[off + d] - m) * rs;
                qa += ln * W_qkv[d*(3u*D) + h*DH + dh];
                ka += ln * W_qkv[d*(3u*D) + D + h*DH + dh];
                va += ln * W_qkv[d*(3u*D) + 2u*D + h*DH + dh];
            }
            Qh[idx] = qa; Kh[idx] = ka; Vh[idx] = va;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float my_sc[4];
        for (uint pi = 0; pi < 4; ++pi) {
            uint pair = t*4 + pi;
            uint sq = pair / S;
            uint kt = pair % S;
            float dot = 0;
            for (uint d = 0; d < DH; ++d) dot += Qh[sq*DH + d] * Kh[kt*DH + d];
            my_sc[pi] = dot * inv_sqrt_dh;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint pi = 0; pi < 4; ++pi) sc[t*4 + pi] = my_sc[pi];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            uint sq = t >> 4; uint lane = t & 15;
            uint base = sq * S;
            float v0 = sc[base + lane];
            float v1 = sc[base + lane + 16];
            float v2 = sc[base + lane + 32];
            float v3 = sc[base + lane + 48];
            float m = max(max(v0,v1), max(v2,v3));
            m = max(m, simd_shuffle_xor(m, 1));
            m = max(m, simd_shuffle_xor(m, 2));
            m = max(m, simd_shuffle_xor(m, 4));
            m = max(m, simd_shuffle_xor(m, 8));
            v0 = fast::exp(v0-m); v1 = fast::exp(v1-m); v2 = fast::exp(v2-m); v3 = fast::exp(v3-m);
            float ss = v0+v1+v2+v3;
            ss += simd_shuffle_xor(ss, 1);
            ss += simd_shuffle_xor(ss, 2);
            ss += simd_shuffle_xor(ss, 4);
            ss += simd_shuffle_xor(ss, 8);
            float inv = 1.0f / ss;
            sc[base + lane]    = v0 * inv;
            sc[base + lane+16] = v1 * inv;
            sc[base + lane+32] = v2 * inv;
            sc[base + lane+48] = v3 * inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float oh_vals[2]; uint oh_idx[2]; uint nn = 0;
        for (uint idx = t; idx < S*DH; idx += TG) {
            uint sq = idx / DH;
            uint dh = idx % DH;
            float acc = 0;
            for (uint k = 0; k < S; ++k) acc += sc[sq*S + k] * Vh[k*DH + dh];
            oh_vals[nn] = acc; oh_idx[nn] = idx; nn++;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < nn; ++i) sc[oh_idx[i]] = oh_vals[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = t; idx < S*D; idx += TG) {
            uint sr = idx / D;
            uint dout = idx % D;
            float acc = 0;
            for (uint dh = 0; dh < DH; ++dh) {
                acc += sc[sr*DH + dh] * W_o[(h*DH + dh)*D + dout];
            }
            if (h == 0) y[sr*D + dout] = x[sr*D + dout] + acc;
            else        y[sr*D + dout] += acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }

    threadgroup float* ln_m = pool;
    threadgroup float* ln_r = pool + 64;
    threadgroup float* hidden = pool + 128;   // size S*FF_BLK = 4096

    {
        const uint row = t >> 4;
        const uint lane = t & 15;
        const uint off = row * D;
        float s1 = 0, s2 = 0;
        for (uint d = lane; d < D; d += 16) {
            float v = y[off + d]; s1 += v; s2 += v*v;
        }
        s1 += simd_shuffle_xor(s1, 1);
        s1 += simd_shuffle_xor(s1, 2);
        s1 += simd_shuffle_xor(s1, 4);
        s1 += simd_shuffle_xor(s1, 8);
        s2 += simd_shuffle_xor(s2, 1);
        s2 += simd_shuffle_xor(s2, 2);
        s2 += simd_shuffle_xor(s2, 4);
        s2 += simd_shuffle_xor(s2, 8);
        if (lane == 0) {
            float m = s1 / float(D);
            float v = max(s2/float(D) - m*m, 0.0f);
            ln_m[row] = m;
            ln_r[row] = rsqrt(v + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;

    float out_acc[8] = {0,0,0,0,0,0,0,0};

    for (uint chunk = 0; chunk < NCHUNKS; ++chunk) {
        const uint f0 = chunk * FF_BLK;

        {
            uint s  = t >> 4;          // 0..63
            uint fb = (t & 15u) * 4;   // 0,4,...,60
            uint f  = f0 + fb;
            float m = ln_m[s], rs = ln_r[s];
            uint yoff = s * D;
            float a0=0, a1=0, a2=0, a3=0;
            for (uint d = 0; d < D; ++d) {
                float ln = (y[yoff + d] - m) * rs;
                float4 w = *((device const float4*)(W_ff1 + d*FF + f));
                a0 += ln * w.x;
                a1 += ln * w.y;
                a2 += ln * w.z;
                a3 += ln * w.w;
            }
            float g0 = 0.5f*a0*(1.0f+precise::tanh(k0*(a0+k1*a0*a0*a0)));
            float g1 = 0.5f*a1*(1.0f+precise::tanh(k0*(a1+k1*a1*a1*a1)));
            float g2 = 0.5f*a2*(1.0f+precise::tanh(k0*(a2+k1*a2*a2*a2)));
            float g3 = 0.5f*a3*(1.0f+precise::tanh(k0*(a3+k1*a3*a3*a3)));
            uint hbase = s * FF_BLK + fb;
            hidden[hbase + 0] = g0;
            hidden[hbase + 1] = g1;
            hidden[hbase + 2] = g2;
            hidden[hbase + 3] = g3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            uint s = t >> 4;
            uint d0 = (t & 15u) * 8;
            uint hoff = s * FF_BLK;
            float o0=0,o1=0,o2=0,o3=0,o4=0,o5=0,o6=0,o7=0;
            for (uint fb = 0; fb < FF_BLK; ++fb) {
                float hv = hidden[hoff + fb];
                uint wb = (f0 + fb) * D + d0;
                float4 wa = *((device const float4*)(W_ff2 + wb));
                float4 wb2 = *((device const float4*)(W_ff2 + wb + 4));
                o0 += hv * wa.x;
                o1 += hv * wa.y;
                o2 += hv * wa.z;
                o3 += hv * wa.w;
                o4 += hv * wb2.x;
                o5 += hv * wb2.y;
                o6 += hv * wb2.z;
                o7 += hv * wb2.w;
            }
            out_acc[0] += o0; out_acc[1] += o1; out_acc[2] += o2; out_acc[3] += o3;
            out_acc[4] += o4; out_acc[5] += o5; out_acc[6] += o6; out_acc[7] += o7;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    {
        uint s = t >> 4;
        uint d0 = (t & 15u) * 8;
        uint base = s * D + d0;
        for (uint k = 0; k < 8; ++k) y[base + k] += out_acc[k];
    }
}
