// llama_decoder_layer (M2) — Fixed attention score computation to avoid simd_shuffle misuse.
// The previous approach used simd_shuffle to broadcast scores per lane, but the logic for
// distributing S=64 scores across 32 lanes was incorrect, causing massive errors.
// Now each thread computes a single score per query position using simd_sum directly.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define S    64
#define D    128
#define H    4
#define HKV  2
#define DH   32
#define G    2
#define FF   256
#define TFF  512
#define TG   1024
#define QKV_W (D + 2*HKV*DH)   // 256
#define HEAD_DIM_F4 (DH/4)     // 8

[[max_total_threads_per_threadgroup(1024)]]
kernel void llama_decoder_layer_f32(
    device const float* x       [[buffer(0)]],
    device const float* W_qkv   [[buffer(1)]],
    device const float* W_o     [[buffer(2)]],
    device const float* W_gu    [[buffer(3)]],
    device const float* W_down  [[buffer(4)]],
    device       float* y       [[buffer(5)]],
    constant uint& S_   [[buffer(6)]],
    constant uint& D_   [[buffer(7)]],
    constant uint& H_   [[buffer(8)]],
    constant uint& HKV_ [[buffer(9)]],
    constant uint& FF_  [[buffer(10)]],
    constant float& base [[buffer(11)]],
    constant float& eps  [[buffer(12)]],
    uint t   [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid_in_sg [[thread_index_in_simdgroup]])
{
    threadgroup float act[S * D];  // 8192 floats = 32KB
    const float log_base = fast::log(base);

    {
        uint row = t >> 4;       // 0..63
        uint lane = t & 15u;     // 0..15
        uint xoff = row * D;
        float ss = 0.0f;
        for (uint d = lane; d < D; d += 16) {
            float v = x[xoff + d];
            ss = fma(v, v, ss);
        }
        ss += simd_shuffle_xor(ss, 1);
        ss += simd_shuffle_xor(ss, 2);
        ss += simd_shuffle_xor(ss, 4);
        ss += simd_shuffle_xor(ss, 8);
        float rstd = rsqrt(ss / float(D) + eps);
        for (uint d = lane; d < D; d += 16) {
            act[xoff + d] = x[xoff + d] * rstd;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        const uint sm = sgid >> 2;       // 0..7
        const uint sn = sgid & 3u;
        const uint row0 = sm * 8;

        const uint qcol0 = sn * 64;  // 0..192 step 64

        simdgroup_matrix<float, 8, 8> C0(0.0f), C1(0.0f), C2(0.0f), C3(0.0f);
        simdgroup_matrix<float, 8, 8> C4(0.0f), C5(0.0f), C6(0.0f), C7(0.0f);
        for (uint kk = 0; kk < 16; ++kk) {
            simdgroup_matrix<float, 8, 8> A_blk;
            simdgroup_load(A_blk, &act[row0 * D + kk*8], D);
            {
                simdgroup_matrix<float, 8, 8> B0, B1, B2, B3;
                simdgroup_load(B0, &W_qkv[(kk*8) * QKV_W + qcol0 + 0],  QKV_W);
                simdgroup_load(B1, &W_qkv[(kk*8) * QKV_W + qcol0 + 8],  QKV_W);
                simdgroup_load(B2, &W_qkv[(kk*8) * QKV_W + qcol0 + 16], QKV_W);
                simdgroup_load(B3, &W_qkv[(kk*8) * QKV_W + qcol0 + 24], QKV_W);
                simdgroup_multiply_accumulate(C0, A_blk, B0, C0);
                simdgroup_multiply_accumulate(C1, A_blk, B1, C1);
                simdgroup_multiply_accumulate(C2, A_blk, B2, C2);
                simdgroup_multiply_accumulate(C3, A_blk, B3, C3);
            }
            {
                simdgroup_matrix<float, 8, 8> B0, B1, B2, B3;
                simdgroup_load(B0, &W_qkv[(kk*8) * QKV_W + qcol0 + 32], QKV_W);
                simdgroup_load(B1, &W_qkv[(kk*8) * QKV_W + qcol0 + 40], QKV_W);
                simdgroup_load(B2, &W_qkv[(kk*8) * QKV_W + qcol0 + 48], QKV_W);
                simdgroup_load(B3, &W_qkv[(kk*8) * QKV_W + qcol0 + 56], QKV_W);
                simdgroup_multiply_accumulate(C4, A_blk, B0, C4);
                simdgroup_multiply_accumulate(C5, A_blk, B1, C5);
                simdgroup_multiply_accumulate(C6, A_blk, B2, C6);
                simdgroup_multiply_accumulate(C7, A_blk, B3, C7);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sn < 2u) {
            uint dst_col = sn * 64;
            simdgroup_store(C0, &y[row0 * D + dst_col + 0],  D);
            simdgroup_store(C1, &y[row0 * D + dst_col + 8],  D);
            simdgroup_store(C2, &y[row0 * D + dst_col + 16], D);
            simdgroup_store(C3, &y[row0 * D + dst_col + 24], D);
            simdgroup_store(C4, &y[row0 * D + dst_col + 32], D);
            simdgroup_store(C5, &y[row0 * D + dst_col + 40], D);
            simdgroup_store(C6, &y[row0 * D + dst_col + 48], D);
            simdgroup_store(C7, &y[row0 * D + dst_col + 56], D);
        } else {
            uint dst_col = (sn - 2u) * 64;
            simdgroup_store(C0, &act[row0 * 128 + dst_col + 0],  128);
            simdgroup_store(C1, &act[row0 * 128 + dst_col + 8],  128);
            simdgroup_store(C2, &act[row0 * 128 + dst_col + 16], 128);
            simdgroup_store(C3, &act[row0 * 128 + dst_col + 24], 128);
            simdgroup_store(C4, &act[row0 * 128 + dst_col + 32], 128);
            simdgroup_store(C5, &act[row0 * 128 + dst_col + 40], 128);
            simdgroup_store(C6, &act[row0 * 128 + dst_col + 48], 128);
            simdgroup_store(C7, &act[row0 * 128 + dst_col + 56], 128);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // RoPE K and Q
    {
        for (uint i = 0; i < 2; ++i) {
            uint pair = t * 2 + i;
            uint s   = pair / (HKV * (DH/2));
            uint rem = pair - s * (HKV * (DH/2));
            uint kv  = rem / (DH/2);
            uint ii  = rem - kv * (DH/2);
            uint col0 = kv * DH + 2u*ii;
            uint col1 = col0 + 1u;
            float k0 = act[s * 128 + col0];
            float k1 = act[s * 128 + col1];
            float omega = fast::exp(-(2.0f * float(ii) / float(DH)) * log_base);
            float ang = float(s) * omega;
            float cv = fast::cos(ang);
            float sv = fast::sin(ang);
            act[s * 128 + col0] = k0 * cv - k1 * sv;
            act[s * 128 + col1] = k0 * sv + k1 * cv;
        }
        for (uint i = 0; i < 4; ++i) {
            uint pair = t * 4 + i;
            uint s   = pair / (H * (DH/2));
            uint rem = pair - s * (H * (DH/2));
            uint hh  = rem / (DH/2);
            uint ii  = rem - hh * (DH/2);
            uint col0 = hh * DH + 2u*ii;
            uint col1 = col0 + 1u;
            float q0 = y[s * D + col0];
            float q1 = y[s * D + col1];
            float omega = fast::exp(-(2.0f * float(ii) / float(DH)) * log_base);
            float ang = float(s) * omega;
            float cv = fast::cos(ang);
            float sv = fast::sin(ang);
            y[s * D + col0] = q0 * cv - q1 * sv;
            y[s * D + col1] = q0 * sv + q1 * cv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Attention: each thread handles one query position and one key-value pair
    // to avoid simd_shuffle broadcasting errors. We iterate over all S keys
    // and accumulate the softmax-weighted value directly.
    {
        const uint sm_a = sgid >> 2;   // 0..7
        const uint sn_a = sgid & 3u;   // 0..3 (head)
        const uint row0 = sm_a * 8;
        const uint head = sn_a;
        const uint kv = head / G;      // 0 or 1

        const float inv_sqrt_dh = rsqrt(float(DH));
        float attn_local[8];

        for (uint r = 0; r < 8; ++r) {
            uint sq = row0 + r;
            float q_lane = y[sq * D + head * DH + lid_in_sg];

            // Compute all S=64 scores for this query position
            float scores[64];
            for (uint kt = 0; kt < S; ++kt) {
                float kv_val = act[kt * 128 + kv * DH + lid_in_sg];
                float prod = q_lane * kv_val;
                // simd_sum across all 32 lanes gives the full dot product
                float dot = simd_sum(prod) * inv_sqrt_dh;
                scores[kt] = dot;
            }

            // Softmax: find max, compute exp, sum
            float m = -INFINITY;
            for (uint kt = 0; kt < S; ++kt) {
                m = max(m, scores[kt]);
            }
            float ssum = 0.0f;
            for (uint kt = 0; kt < S; ++kt) {
                ssum += fast::exp(scores[kt] - m);
            }
            float inv = 1.0f / ssum;
            float ao = 0.0f;
            for (uint kt = 0; kt < S; ++kt) {
                float p = fast::exp(scores[kt] - m) * inv;
                float vval = act[kt * 128 + 64u + kv * DH + lid_in_sg];
                ao = fma(p, vval, ao);
            }
            attn_local[r] = ao;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < 8; ++r) {
            act[(row0 + r) * D + head * DH + lid_in_sg] = attn_local[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Output projection
    {
        const uint sm_a = sgid >> 2;
        const uint sn_a = sgid & 3u;
        const uint row0 = sm_a * 8;
        const uint col0 = sn_a * 32;

        simdgroup_matrix<float, 8, 8> C0(0.0f), C1(0.0f), C2(0.0f), C3(0.0f);

        for (uint kc = 0; kc < D; kc += 8) {
            simdgroup_matrix<float, 8, 8> A_blk;
            simdgroup_matrix<float, 8, 8> B0, B1, B2, B3;
            simdgroup_load(A_blk, &act[row0 * D + kc], D);
            simdgroup_load(B0, &W_o[kc * D + col0 + 0],  D);
            simdgroup_load(B1, &W_o[kc * D + col0 + 8],  D);
            simdgroup_load(B2, &W_o[kc * D + col0 + 16], D);
            simdgroup_load(B3, &W_o[kc * D + col0 + 24], D);
            simdgroup_multiply_accumulate(C0, A_blk, B0, C0);
            simdgroup_multiply_accumulate(C1, A_blk, B1, C1);
            simdgroup_multiply_accumulate(C2, A_blk, B2, C2);
            simdgroup_multiply_accumulate(C3, A_blk, B3, C3);
        }
        simdgroup_store(C0, &y[row0 * D + col0 + 0],  D);
        simdgroup_store(C1, &y[row0 * D + col0 + 8],  D);
        simdgroup_store(C2, &y[row0 * D + col0 + 16], D);
        simdgroup_store(C3, &y[row0 * D + col0 + 24], D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float ffn_acc[8];
    {
        for (uint i = 0; i < 8; ++i) {
            uint cell = t * 8 + i;
            float v = y[cell] + x[cell];
            ffn_acc[i] = v;
            act[cell] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Second RMSNorm
    {
        uint row = t >> 4;
        uint lane = t & 15u;
        uint off = row * D;
        float ss = 0.0f;
        for (uint d = lane; d < D; d += 16) {
            float v = act[off + d];
            ss = fma(v, v, ss);
        }
        ss += simd_shuffle_xor(ss, 1);
        ss += simd_shuffle_xor(ss, 2);
        ss += simd_shuffle_xor(ss, 4);
        ss += simd_shuffle_xor(ss, 8);
        float rstd = rsqrt(ss / float(D) + eps);
        for (uint d = lane; d < D; d += 16) {
            act[off + d] *= rstd;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint FFC = 64;
    const uint NCHUNKS = FF / FFC;  // 4

    simdgroup_matrix<float, 8, 8> Cff0(0.0f), Cff1(0.0f), Cff2(0.0f), Cff3(0.0f);

    const uint sm_p = sgid >> 2;
    const uint sn_p = sgid & 3u;
    const uint row0_p = sm_p * 8;
    const uint col0_p = sn_p * 32;

    for (uint cc = 0; cc < NCHUNKS; ++cc) {
        const uint fc = cc * FFC;

        {
            const uint row0 = row0_p;
            const uint col0 = col0_p;
            simdgroup_matrix<float, 8, 8> C0(0.0f), C1(0.0f), C2(0.0f), C3(0.0f);
            uint w_col_base;
            if (col0 < 64u) w_col_base = fc + col0;
            else            w_col_base = FF + fc + (col0 - 64u);

            for (uint kc = 0; kc < D; kc += 8) {
                simdgroup_matrix<float, 8, 8> A_blk;
                simdgroup_matrix<float, 8, 8> B0, B1, B2, B3;
                simdgroup_load(A_blk, &act[row0 * D + kc], D);
                simdgroup_load(B0, &W_gu[kc * TFF + w_col_base + 0],  TFF);
                simdgroup_load(B1, &W_gu[kc * TFF + w_col_base + 8],  TFF);
                simdgroup_load(B2, &W_gu[kc * TFF + w_col_base + 16], TFF);
                simdgroup_load(B3, &W_gu[kc * TFF + w_col_base + 24], TFF);
                simdgroup_multiply_accumulate(C0, A_blk, B0, C0);
                simdgroup_multiply_accumulate(C1, A_blk, B1, C1);
                simdgroup_multiply_accumulate(C2, A_blk, B2, C2);
                simdgroup_multiply_accumulate(C3, A_blk, B3, C3);
            }
            simdgroup_store(C0, &y[row0 * 128 + col0 + 0],  128);
            simdgroup_store(C1, &y[row0 * 128 + col0 + 8],  128);
            simdgroup_store(C2, &y[row0 * 128 + col0 + 16], 128);
            simdgroup_store(C3, &y[row0 * 128 + col0 + 24], 128);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            for (uint i = 0; i < 4; ++i) {
                uint cell = t * 4 + i;
                uint s = cell >> 6;
                uint fb = cell & 63u;
                float g = y[s * 128 + fb];
                float u = y[s * 128 + 64 + fb];
                float sg = g / (1.0f + fast::exp(-g));
                y[s * 128 + fb] = sg * u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            const uint row0 = row0_p;
            const uint col0 = col0_p;
            for (uint kc = 0; kc < FFC; kc += 8) {
                simdgroup_matrix<float, 8, 8> A_blk;
                simdgroup_matrix<float, 8, 8> B0, B1, B2, B3;
                simdgroup_load(A_blk, &y[row0 * 128 + kc], 128);
                simdgroup_load(B0, &W_down[(fc + kc) * D + col0 + 0],  D);
                simdgroup_load(B1, &W_down[(fc + kc) * D + col0 + 8],  D);
                simdgroup_load(B2, &W_down[(fc + kc) * D + col0 + 16], D);
                simdgroup_load(B3, &W_down[(fc + kc) * D + col0 + 24], D);
                simdgroup_multiply_accumulate(Cff0, A_blk, B0, Cff0);
                simdgroup_multiply_accumulate(Cff1, A_blk, B1, Cff1);
                simdgroup_multiply_accumulate(Cff2, A_blk, B2, Cff2);
                simdgroup_multiply_accumulate(Cff3, A_blk, B3, Cff3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    {
        simdgroup_store(Cff0, &y[row0_p * D + col0_p + 0],  D);
        simdgroup_store(Cff1, &y[row0_p * D + col0_p + 8],  D);
        simdgroup_store(Cff2, &y[row0_p * D + col0_p + 16], D);
        simdgroup_store(Cff3, &y[row0_p * D + col0_p + 24], D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        for (uint i = 0; i < 8; ++i) {
            y[t * 8 + i] += ffn_acc[i];
        }
    }
}
