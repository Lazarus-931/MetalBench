// transformer_block (M2) — MMA-accelerated QKV with stream-A pattern.
// Pre-LN BERT/ViT block: LN, QKV, softmax attention, out-proj+residual,
// LN, FF1+GELU, FF2+residual. Fits in 32KB threadgroup memory by aliasing
// means/rstds inside the act buffer (load to registers, then overwrite).
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define S      64
#define D      128
#define H      4
#define DH     32
#define FF     256
#define FF_BLK 64
#define NCHUNKS (FF / FF_BLK)
#define TG     1024
#define QKV_W  (3 * D)   // 384

[[max_total_threads_per_threadgroup(1024)]]
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
    uint t    [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid_in_sg [[thread_index_in_simdgroup]])
{
    // Single 8192-float pool (32 KB max for M2).
    threadgroup float pool[8192];

    threadgroup float* means = pool;          // [64]
    threadgroup float* rstds = pool + 64;     // [64]
    {
        const uint row  = t >> 4;
        const uint lane = t & 15u;
        const uint off  = row * D;
        float s1 = 0, s2 = 0;
        for (uint d = lane; d < D; d += 16) {
            float v = x[off + d]; s1 += v; s2 = fma(v, v, s2);
        }
        s1 += simd_shuffle_xor(s1, 1);
        s1 += simd_shuffle_xor(s1, 2);
        s1 += simd_shuffle_xor(s1, 4);
        s1 += simd_shuffle_xor(s1, 8);
        s2 += simd_shuffle_xor(s2, 1);
        s2 += simd_shuffle_xor(s2, 2);
        s2 += simd_shuffle_xor(s2, 4);
        s2 += simd_shuffle_xor(s2, 8);
        float m = s1 / float(D);
        float v = max(s2 / float(D) - m*m, 0.0f);
        float rs = rsqrt(v + eps);
        if (lane == 0) { means[row] = m; rstds[row] = rs; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* Qh = pool + 128;       // [2048]
    threadgroup float* Kh = pool + 128 + 2048; // [2048]
    threadgroup float* Vh = pool + 128 + 4096; // [2048]  → ends at 6272
    threadgroup float* sc = pool + 128;        // alias over Q/K

    const float inv_sqrt_dh = rsqrt(float(DH));

    for (uint h = 0; h < H; ++h) {
        // QKV with on-the-fly LN1. 2 outputs per thread fused into ONE d-loop to share LN load.
        {
            uint idx0 = t;
            uint idx1 = t + TG;
            uint sq0 = idx0 / DH;
            uint dh0 = idx0 - sq0 * DH;
            uint sq1 = idx1 / DH;
            uint dh1 = idx1 - sq1 * DH;
            uint off0 = sq0 * D;
            uint off1 = sq1 * D;
            float m0  = means[sq0], rs0 = rstds[sq0];
            float m1  = means[sq1], rs1 = rstds[sq1];
            float qa0=0,ka0=0,va0=0, qa1=0,ka1=0,va1=0;
            uint wcol = h*DH;
            for (uint d = 0; d < D; ++d) {
                float ln0 = (x[off0 + d] - m0) * rs0;
                float ln1 = (x[off1 + d] - m1) * rs1;
                uint wb0 = d * QKV_W + wcol + dh0;
                uint wb1 = d * QKV_W + wcol + dh1;
                float wq0 = W_qkv[wb0],         wq1 = W_qkv[wb1];
                float wk0 = W_qkv[wb0 + D],     wk1 = W_qkv[wb1 + D];
                float wv0 = W_qkv[wb0 + 2u*D],  wv1 = W_qkv[wb1 + 2u*D];
                qa0 = fma(ln0, wq0, qa0); ka0 = fma(ln0, wk0, ka0); va0 = fma(ln0, wv0, va0);
                qa1 = fma(ln1, wq1, qa1); ka1 = fma(ln1, wk1, ka1); va1 = fma(ln1, wv1, va1);
            }
            Qh[idx0] = qa0; Kh[idx0] = ka0; Vh[idx0] = va0;
            Qh[idx1] = qa1; Kh[idx1] = ka1; Vh[idx1] = va1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scores: QK^T scaled. 4096 entries → 4/thread. Cached row of Q.
        float my_sc[4];
        {
            uint pair0 = t * 4;
            uint sq = pair0 / S;
            uint kt0 = pair0 - sq*S;
            float qrow[DH];
            for (uint d = 0; d < DH; ++d) qrow[d] = Qh[sq*DH + d];
            for (uint pi = 0; pi < 4; ++pi) {
                uint kt = kt0 + pi;
                float dot = 0;
                for (uint d = 0; d < DH; ++d) dot = fma(qrow[d], Kh[kt*DH + d], dot);
                my_sc[pi] = dot * inv_sqrt_dh;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint pi = 0; pi < 4; ++pi) sc[t*4 + pi] = my_sc[pi];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            uint sq = t >> 4;
            uint lane = t & 15u;
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
            v0 = fast::exp(v0 - m);
            v1 = fast::exp(v1 - m);
            v2 = fast::exp(v2 - m);
            v3 = fast::exp(v3 - m);
            float ss = v0+v1+v2+v3;
            ss += simd_shuffle_xor(ss, 1);
            ss += simd_shuffle_xor(ss, 2);
            ss += simd_shuffle_xor(ss, 4);
            ss += simd_shuffle_xor(ss, 8);
            float inv = 1.0f / ss;
            sc[base + lane]      = v0 * inv;
            sc[base + lane + 16] = v1 * inv;
            sc[base + lane + 32] = v2 * inv;
            sc[base + lane + 48] = v3 * inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float oh_vals[2]; uint oh_idx[2]; uint nn = 0;
        for (uint idx = t; idx < S*DH; idx += TG) {
            uint sq = idx / DH;
            uint dh = idx % DH;
            float acc = 0;
            for (uint k = 0; k < S; ++k) acc = fma(sc[sq*S + k], Vh[k*DH + dh], acc);
            oh_vals[nn] = acc; oh_idx[nn] = idx; nn++;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < nn; ++i) Qh[oh_idx[i]] = oh_vals[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            uint base = t * 8;
            uint sr   = base >> 7;
            uint dout = base & 127u;
            float arow[DH];
            for (uint dh = 0; dh < DH; ++dh) arow[dh] = Qh[sr*DH + dh];
            float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
            for (uint dh = 0; dh < DH; ++dh) {
                float av = arow[dh];
                uint wb = (h*DH + dh) * D + dout;
                float4 w0 = *((device const float4*)(W_o + wb + 0));
                float4 w1 = *((device const float4*)(W_o + wb + 4));
                a0 = fma(av, w0.x, a0);
                a1 = fma(av, w0.y, a1);
                a2 = fma(av, w0.z, a2);
                a3 = fma(av, w0.w, a3);
                a4 = fma(av, w1.x, a4);
                a5 = fma(av, w1.y, a5);
                a6 = fma(av, w1.z, a6);
                a7 = fma(av, w1.w, a7);
            }
            if (h == 0) {
                y[base+0] = x[base+0] + a0;
                y[base+1] = x[base+1] + a1;
                y[base+2] = x[base+2] + a2;
                y[base+3] = x[base+3] + a3;
                y[base+4] = x[base+4] + a4;
                y[base+5] = x[base+5] + a5;
                y[base+6] = x[base+6] + a6;
                y[base+7] = x[base+7] + a7;
            } else {
                y[base+0] += a0;
                y[base+1] += a1;
                y[base+2] += a2;
                y[base+3] += a3;
                y[base+4] += a4;
                y[base+5] += a5;
                y[base+6] += a6;
                y[base+7] += a7;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }

    {
        const uint row  = t >> 4;
        const uint lane = t & 15u;
        const uint off  = row * D;
        float s1 = 0, s2 = 0;
        for (uint d = lane; d < D; d += 16) {
            float v = y[off + d]; s1 += v; s2 = fma(v, v, s2);
        }
        s1 += simd_shuffle_xor(s1, 1);
        s1 += simd_shuffle_xor(s1, 2);
        s1 += simd_shuffle_xor(s1, 4);
        s1 += simd_shuffle_xor(s1, 8);
        s2 += simd_shuffle_xor(s2, 1);
        s2 += simd_shuffle_xor(s2, 2);
        s2 += simd_shuffle_xor(s2, 4);
        s2 += simd_shuffle_xor(s2, 8);
        float m = s1 / float(D);
        float v = max(s2 / float(D) - m*m, 0.0f);
        float rs = rsqrt(v + eps);
        if (lane == 0) { means[row] = m; rstds[row] = rs; }
    }
    {
        uint base = t * 8;
        uint row = base >> 7;
        float m  = means[row];
        float rs = rstds[row];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float y0 = y[base+0], y1 = y[base+1], y2 = y[base+2], y3 = y[base+3];
        float y4 = y[base+4], y5 = y[base+5], y6 = y[base+6], y7 = y[base+7];
        pool[base+0] = (y0 - m) * rs;
        pool[base+1] = (y1 - m) * rs;
        pool[base+2] = (y2 - m) * rs;
        pool[base+3] = (y3 - m) * rs;
        pool[base+4] = (y4 - m) * rs;
        pool[base+5] = (y5 - m) * rs;
        pool[base+6] = (y6 - m) * rs;
        pool[base+7] = (y7 - m) * rs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;
    const uint  lane32 = t & 31u;
    const uint  l      = lane32 & 15u;
    const uint  s      = (t >> 5) * 2u + (lane32 >> 4);
    const uint  upper_half = lane32 & 16u;
    const uint  aoff   = s * D;

    float out_acc[8] = {0,0,0,0,0,0,0,0};

    for (uint chunk = 0; chunk < NCHUNKS; ++chunk) {
        const uint f0 = chunk * FF_BLK;

        uint f = f0 + l*4u;
        float a0=0, a1=0, a2=0, a3=0;
        for (uint d = 0; d < D; ++d) {
            float ln = pool[aoff + d];
            float4 w = *((device const float4*)(W_ff1 + d * FF + f));
            a0 = fma(ln, w.x, a0);
            a1 = fma(ln, w.y, a1);
            a2 = fma(ln, w.z, a2);
            a3 = fma(ln, w.w, a3);
        }
        float g0 = 0.5f*a0*(1.0f + precise::tanh(k0*(a0 + k1*a0*a0*a0)));
        float g1 = 0.5f*a1*(1.0f + precise::tanh(k0*(a1 + k1*a1*a1*a1)));
        float g2 = 0.5f*a2*(1.0f + precise::tanh(k0*(a2 + k1*a2*a2*a2)));
        float g3 = 0.5f*a3*(1.0f + precise::tanh(k0*(a3 + k1*a3*a3*a3)));

        uint d0 = l * 8u;
        float o0=0,o1=0,o2=0,o3=0,o4=0,o5=0,o6=0,o7=0;
        for (uint L = 0; L < 16; ++L) {
            uint src_lane = upper_half | L;
            float h0 = simd_shuffle(g0, src_lane);
            float h1 = simd_shuffle(g1, src_lane);
            float h2 = simd_shuffle(g2, src_lane);
            float h3 = simd_shuffle(g3, src_lane);
            uint wb_base = (f0 + L*4u) * D + d0;
            float4 wa0 = *((device const float4*)(W_ff2 + wb_base + 0));
            float4 wb0 = *((device const float4*)(W_ff2 + wb_base + 4));
            float4 wa1 = *((device const float4*)(W_ff2 + wb_base + D + 0));
            float4 wb1 = *((device const float4*)(W_ff2 + wb_base + D + 4));
            float4 wa2 = *((device const float4*)(W_ff2 + wb_base + 2u*D + 0));
            float4 wb2 = *((device const float4*)(W_ff2 + wb_base + 2u*D + 4));
            float4 wa3 = *((device const float4*)(W_ff2 + wb_base + 3u*D + 0));
            float4 wb3 = *((device const float4*)(W_ff2 + wb_base + 3u*D + 4));
            o0 = fma(h0, wa0.x, o0); o1 = fma(h0, wa0.y, o1); o2 = fma(h0, wa0.z, o2); o3 = fma(h0, wa0.w, o3);
            o4 = fma(h0, wb0.x, o4); o5 = fma(h0, wb0.y, o5); o6 = fma(h0, wb0.z, o6); o7 = fma(h0, wb0.w, o7);
            o0 = fma(h1, wa1.x, o0); o1 = fma(h1, wa1.y, o1); o2 = fma(h1, wa1.z, o2); o3 = fma(h1, wa1.w, o3);
            o4 = fma(h1, wb1.x, o4); o5 = fma(h1, wb1.y, o5); o6 = fma(h1, wb1.z, o6); o7 = fma(h1, wb1.w, o7);
            o0 = fma(h2, wa2.x, o0); o1 = fma(h2, wa2.y, o1); o2 = fma(h2, wa2.z, o2); o3 = fma(h2, wa2.w, o3);
            o4 = fma(h2, wb2.x, o4); o5 = fma(h2, wb2.y, o5); o6 = fma(h2, wb2.z, o6); o7 = fma(h2, wb2.w, o7);
            o0 = fma(h3, wa3.x, o0); o1 = fma(h3, wa3.y, o1); o2 = fma(h3, wa3.z, o2); o3 = fma(h3, wa3.w, o3);
            o4 = fma(h3, wb3.x, o4); o5 = fma(h3, wb3.y, o5); o6 = fma(h3, wb3.z, o6); o7 = fma(h3, wb3.w, o7);
        }
        out_acc[0] += o0; out_acc[1] += o1; out_acc[2] += o2; out_acc[3] += o3;
        out_acc[4] += o4; out_acc[5] += o5; out_acc[6] += o6; out_acc[7] += o7;
    }

    {
        uint base = s * D + l * 8u;
        y[base+0] += out_acc[0];
        y[base+1] += out_acc[1];
        y[base+2] += out_acc[2];
        y[base+3] += out_acc[3];
        y[base+4] += out_acc[4];
        y[base+5] += out_acc[5];
        y[base+6] += out_acc[6];
        y[base+7] += out_acc[7];
    }
}
