// llama_decoder_layer (M4) — MMA-based redesign. Single TG of 1024 threads.
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

        simdgroup_matrix<float, 8, 8> A0,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15;
        simdgroup_load(A0,  &act[row0 * D + 0*8], D);
        simdgroup_load(A1,  &act[row0 * D + 1*8], D);
        simdgroup_load(A2,  &act[row0 * D + 2*8], D);
        simdgroup_load(A3,  &act[row0 * D + 3*8], D);
        simdgroup_load(A4,  &act[row0 * D + 4*8], D);
        simdgroup_load(A5,  &act[row0 * D + 5*8], D);
        simdgroup_load(A6,  &act[row0 * D + 6*8], D);
        simdgroup_load(A7,  &act[row0 * D + 7*8], D);
        simdgroup_load(A8,  &act[row0 * D + 8*8], D);
        simdgroup_load(A9,  &act[row0 * D + 9*8], D);
        simdgroup_load(A10, &act[row0 * D + 10*8], D);
        simdgroup_load(A11, &act[row0 * D + 11*8], D);
        simdgroup_load(A12, &act[row0 * D + 12*8], D);
        simdgroup_load(A13, &act[row0 * D + 13*8], D);
        simdgroup_load(A14, &act[row0 * D + 14*8], D);
        simdgroup_load(A15, &act[row0 * D + 15*8], D);

        threadgroup_barrier(mem_flags::mem_threadgroup);  // ensure all simdgroups finished reading act

        // Compute full QKV: each simdgroup handles 8 rows x 64 cols = 8 tiles.
        // col block: sn=0..3 → output cols sn*64 .. sn*64+63 of QKV_W=256.
        // cols 0..127 = Q (write to y), cols 128..255 = K|V (write to act).
        simdgroup_matrix<float, 8, 8> C0(0.0f), C1(0.0f), C2(0.0f), C3(0.0f),
                                     C4(0.0f), C5(0.0f), C6(0.0f), C7(0.0f);
        const uint qcol0 = sn * 64;  // 0..192 step 64
        #define DO_K(IDX) { \
            simdgroup_matrix<float, 8, 8> B0, B1, B2, B3, B4, B5, B6, B7; \
            simdgroup_load(B0, &W_qkv[(IDX*8) * QKV_W + qcol0 + 0],  QKV_W); \
            simdgroup_load(B1, &W_qkv[(IDX*8) * QKV_W + qcol0 + 8],  QKV_W); \
            simdgroup_load(B2, &W_qkv[(IDX*8) * QKV_W + qcol0 + 16], QKV_W); \
            simdgroup_load(B3, &W_qkv[(IDX*8) * QKV_W + qcol0 + 24], QKV_W); \
            simdgroup_load(B4, &W_qkv[(IDX*8) * QKV_W + qcol0 + 32], QKV_W); \
            simdgroup_load(B5, &W_qkv[(IDX*8) * QKV_W + qcol0 + 40], QKV_W); \
            simdgroup_load(B6, &W_qkv[(IDX*8) * QKV_W + qcol0 + 48], QKV_W); \
            simdgroup_load(B7, &W_qkv[(IDX*8) * QKV_W + qcol0 + 56], QKV_W); \
            simdgroup_multiply_accumulate(C0, A##IDX, B0, C0); \
            simdgroup_multiply_accumulate(C1, A##IDX, B1, C1); \
            simdgroup_multiply_accumulate(C2, A##IDX, B2, C2); \
            simdgroup_multiply_accumulate(C3, A##IDX, B3, C3); \
            simdgroup_multiply_accumulate(C4, A##IDX, B4, C4); \
            simdgroup_multiply_accumulate(C5, A##IDX, B5, C5); \
            simdgroup_multiply_accumulate(C6, A##IDX, B6, C6); \
            simdgroup_multiply_accumulate(C7, A##IDX, B7, C7); \
        }
        DO_K(0) DO_K(1) DO_K(2) DO_K(3) DO_K(4) DO_K(5) DO_K(6) DO_K(7)
        DO_K(8) DO_K(9) DO_K(10) DO_K(11) DO_K(12) DO_K(13) DO_K(14) DO_K(15)
        #undef DO_K

        // Lower 2 simdgroup_matrices (C0..C3, cols 0..31 of qcol0 block) and
        // upper 2 (C4..C7, cols 32..63). For sn=0,1 (qcol0=0,64) all 64 cols
        // are Q → store to y. For sn=2,3 (qcol0=128,192) all 64 cols are KV →
        // store to act (K|V buffer of width 128).
        if (sn < 2u) {
            uint dst_col = sn * 64;  // 0 or 64 within Q
            simdgroup_store(C0, &y[row0 * D + dst_col + 0],  D);
            simdgroup_store(C1, &y[row0 * D + dst_col + 8],  D);
            simdgroup_store(C2, &y[row0 * D + dst_col + 16], D);
            simdgroup_store(C3, &y[row0 * D + dst_col + 24], D);
            simdgroup_store(C4, &y[row0 * D + dst_col + 32], D);
            simdgroup_store(C5, &y[row0 * D + dst_col + 40], D);
            simdgroup_store(C6, &y[row0 * D + dst_col + 48], D);
            simdgroup_store(C7, &y[row0 * D + dst_col + 56], D);
        } else {
            uint dst_col = (sn - 2u) * 64;  // 0 or 64 within K|V (width 128)
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
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // RoPE K (in act) and Q (in y device buf). Each thread handles 2 pairs.
    // For K: HKV*DH/2 = 2*16 = 32 pairs per row → 32*S = 2048 pairs total.
    // For Q: H*DH/2 = 4*16 = 64 pairs per row → 64*S = 4096 pairs total.
    // Total 6144 pairs / 1024 threads = 6 pairs per thread.
    {
        // K rotation: 2048 pairs, ~2 per thread for first 1024 threads work,
        // but we have exactly t in 0..1023. Use t < 2048/2 = 1024 → 2 pairs each
        // covers all K, since 1024*2 = 2048.
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
        // Q rotation: 4096 pairs / 1024 threads = 4 per thread.
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
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    #if 0
    {
        const uint sm = sgid >> 2;
        const uint sn = sgid & 3u;
        const uint row0 = sm * 8;
        const uint col0 = sn * 32;

        simdgroup_matrix<float, 8, 8> C0(0.0f), C1(0.0f), C2(0.0f), C3(0.0f);

    }
    #endif

    {
        const uint sm_a = sgid >> 2;   // 0..7
        const uint sn_a = sgid & 3u;   // 0..3 (head)
        const uint row0 = sm_a * 8;
        const uint head = sn_a;
        const uint kv = head / G;      // 0 or 1

        // Q already computed via QKV MMA and RoPE'd into y. Load it.
        float q_local[8];
        const uint dh_lane = lid_in_sg;
        for (uint r = 0; r < 8; ++r) {
            uint sq = row0 + r;
            q_local[r] = y[sq * D + head * DH + dh_lane];
        }

        const float inv_sqrt_dh = rsqrt(float(DH));
        float scores[8][2];  // 8 rows × 64 kt = too many. Store 64 kt: per lane, 64/32 = 2 kt values.

        // Loop kt outer so K is loaded once per kt, reused across 8 rows.
        #pragma clang loop unroll(disable)
        for (uint kt = 0; kt < S; ++kt) {
            float kv_val = act[kt * 128 + kv * DH + dh_lane];
            uint sel_lane = kt >> 1;
            uint sel_idx  = kt & 1u;
            #pragma clang loop unroll(full)
            for (uint r = 0; r < 8; ++r) {
                float prod = q_local[r] * kv_val;
                float dot = simd_sum(prod) * inv_sqrt_dh;
                if (lid_in_sg == sel_lane) {
                    scores[r][sel_idx] = dot;
                }
            }
        }

        for (uint r = 0; r < 8; ++r) {
            float v0 = (lid_in_sg < 32u) ? scores[r][0] : -INFINITY;
            float v1 = (lid_in_sg < 32u) ? scores[r][1] : -INFINITY;
            float m = max(v0, v1);
            m = max(m, simd_shuffle_xor(m, 1));
            m = max(m, simd_shuffle_xor(m, 2));
            m = max(m, simd_shuffle_xor(m, 4));
            m = max(m, simd_shuffle_xor(m, 8));
            m = max(m, simd_shuffle_xor(m, 16));
            float e0 = fast::exp(v0 - m);
            float e1 = fast::exp(v1 - m);
            float ssum = e0 + e1;
            ssum = simd_sum(ssum);
            float inv = 1.0f / ssum;
            scores[r][0] = e0 * inv;
            scores[r][1] = e1 * inv;
        }

        float attn_local[8] = {0,0,0,0,0,0,0,0};
        #pragma clang loop unroll(disable)
        for (uint kt = 0; kt < S; ++kt) {
            float vval = act[kt * 128 + 64u + kv * DH + dh_lane];
            uint src_lane = kt >> 1;
            uint sel_idx  = kt & 1u;
            #pragma clang loop unroll(full)
            for (uint r = 0; r < 8; ++r) {
                float p = (sel_idx == 0u) ? scores[r][0] : scores[r][1];
                float p_bcast = simd_shuffle(p, src_lane);
                attn_local[r] = fma(p_bcast, vval, attn_local[r]);
            }
        }
        (void)dh_lane;

        threadgroup_barrier(mem_flags::mem_threadgroup);  // all simdgroups finish using K,V.

        for (uint r = 0; r < 8; ++r) {
            act[(row0 + r) * D + head * DH + dh_lane] = attn_local[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

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

    const uint FFC = 64;       // chunk size along FF
    const uint NCHUNKS = FF / FFC;  // 4

    simdgroup_matrix<float, 8, 8> Cff0(0.0f), Cff1(0.0f), Cff2(0.0f), Cff3(0.0f);

    const uint sm_p = sgid >> 2;
    const uint sn_p = sgid & 3u;
    const uint row0_p = sm_p * 8;
    const uint col0_p = sn_p * 32;

    // Preload A (normalized input, 8 rows × 128 cols = 16 tiles) once for reuse
    // across all 4 W_gu chunks.
    simdgroup_matrix<float, 8, 8> Apre0, Apre1, Apre2, Apre3, Apre4, Apre5, Apre6,
                                  Apre7, Apre8, Apre9, Apre10, Apre11, Apre12,
                                  Apre13, Apre14, Apre15;
    simdgroup_load(Apre0,  &act[row0_p * D + 0*8],  D);
    simdgroup_load(Apre1,  &act[row0_p * D + 1*8],  D);
    simdgroup_load(Apre2,  &act[row0_p * D + 2*8],  D);
    simdgroup_load(Apre3,  &act[row0_p * D + 3*8],  D);
    simdgroup_load(Apre4,  &act[row0_p * D + 4*8],  D);
    simdgroup_load(Apre5,  &act[row0_p * D + 5*8],  D);
    simdgroup_load(Apre6,  &act[row0_p * D + 6*8],  D);
    simdgroup_load(Apre7,  &act[row0_p * D + 7*8],  D);
    simdgroup_load(Apre8,  &act[row0_p * D + 8*8],  D);
    simdgroup_load(Apre9,  &act[row0_p * D + 9*8],  D);
    simdgroup_load(Apre10, &act[row0_p * D + 10*8], D);
    simdgroup_load(Apre11, &act[row0_p * D + 11*8], D);
    simdgroup_load(Apre12, &act[row0_p * D + 12*8], D);
    simdgroup_load(Apre13, &act[row0_p * D + 13*8], D);
    simdgroup_load(Apre14, &act[row0_p * D + 14*8], D);
    simdgroup_load(Apre15, &act[row0_p * D + 15*8], D);

    for (uint cc = 0; cc < NCHUNKS; ++cc) {
        const uint fc = cc * FFC;

        {
            const uint row0 = row0_p;
            const uint col0 = col0_p;
            simdgroup_matrix<float, 8, 8> C0(0.0f), C1(0.0f), C2(0.0f), C3(0.0f);
            uint w_col_base;
            if (col0 < 64u) w_col_base = fc + col0;
            else            w_col_base = FF + fc + (col0 - 64u);

            #define WGU_K(IDX) { \
                simdgroup_matrix<float, 8, 8> B0, B1, B2, B3; \
                simdgroup_load(B0, &W_gu[(IDX*8) * TFF + w_col_base + 0],  TFF); \
                simdgroup_load(B1, &W_gu[(IDX*8) * TFF + w_col_base + 8],  TFF); \
                simdgroup_load(B2, &W_gu[(IDX*8) * TFF + w_col_base + 16], TFF); \
                simdgroup_load(B3, &W_gu[(IDX*8) * TFF + w_col_base + 24], TFF); \
                simdgroup_multiply_accumulate(C0, Apre##IDX, B0, C0); \
                simdgroup_multiply_accumulate(C1, Apre##IDX, B1, C1); \
                simdgroup_multiply_accumulate(C2, Apre##IDX, B2, C2); \
                simdgroup_multiply_accumulate(C3, Apre##IDX, B3, C3); \
            }
            WGU_K(0) WGU_K(1) WGU_K(2) WGU_K(3) WGU_K(4) WGU_K(5) WGU_K(6) WGU_K(7)
            WGU_K(8) WGU_K(9) WGU_K(10) WGU_K(11) WGU_K(12) WGU_K(13) WGU_K(14) WGU_K(15)
            #undef WGU_K
            // Stage gate||up in act (threadgroup) instead of y (device).
            simdgroup_store(C0, &act[row0 * 128 + col0 + 0],  128);
            simdgroup_store(C1, &act[row0 * 128 + col0 + 8],  128);
            simdgroup_store(C2, &act[row0 * 128 + col0 + 16], 128);
            simdgroup_store(C3, &act[row0 * 128 + col0 + 24], 128);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            for (uint i = 0; i < 4; ++i) {
                uint cell = t * 4 + i;
                uint s = cell >> 6;
                uint fb = cell & 63u;
                float g = act[s * 128 + fb];
                float u = act[s * 128 + 64 + fb];
                float sg = g / (1.0f + fast::exp(-g));
                act[s * 128 + fb] = sg * u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            const uint row0 = row0_p;
            const uint col0 = col0_p;
            for (uint kc = 0; kc < FFC; kc += 8) {
                simdgroup_matrix<float, 8, 8> A_blk;
                simdgroup_matrix<float, 8, 8> B0, B1, B2, B3;
                simdgroup_load(A_blk, &act[row0 * 128 + kc], 128);
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
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    {
        for (uint i = 0; i < 8; ++i) {
            y[t * 8 + i] += ffn_acc[i];
        }
    }
}
