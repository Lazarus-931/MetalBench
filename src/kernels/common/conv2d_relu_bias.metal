// conv2d_relu_bias: fused 3x3 conv (NHWC) + bias + ReLU, NHWC f32.
// Implicit-im2col GEMM via simdgroup_matrix MMA. Each TG processes 2 consecutive
// m-tiles, sharing per-kt B (weight) load → halves B bandwidth and reuses each
// loaded B-block across two simdgroup_matrix accumulators.
// Grid-agnostic: each TG strides over its share of m-tile groups via tg_id.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM = 64, BN = 128, BK = 16;
constant constexpr uint SM = 16, SN = 16;
constant constexpr uint SIMDS_M = BM / SM;   // 4
constant constexpr uint SIMDS_N = BN / SN;   // 8
constant constexpr uint MMA_M = SM / 8;      // 2
constant constexpr uint MMA_N = SN / 8;      // 2
constant constexpr uint TG_THREADS = 1024;
constant constexpr uint PAD = 4;
constant constexpr uint LDA = BK + PAD;      // 20
constant constexpr uint LDB = BN + PAD;      // 132

kernel void conv2d_relu_bias_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  w       [[buffer(1)]],
    device const float*  b       [[buffer(2)]],
    device       float*  y       [[buffer(3)]],
    constant     uint&   N_      [[buffer(4)]],
    constant     uint&   H       [[buffer(5)]],
    constant     uint&   W       [[buffer(6)]],
    constant     uint&   C       [[buffer(7)]],
    constant     uint&   K_out   [[buffer(8)]],
    constant     uint&   R       [[buffer(9)]],
    constant     uint&   S       [[buffer(10)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint sgid    [[simdgroup_index_in_threadgroup]],
    uint lid     [[thread_index_in_threadgroup]],
    uint n_tg    [[threadgroups_per_grid]])
{
    const uint H2 = (H - R) + 1;
    const uint W2 = (W - S) + 1;
    const uint HW2 = H2 * W2;
    const uint M_g = N_ * HW2;
    const uint K_g = R * S * C;
    const uint SC  = S * C;
    const uint num_m_tiles = (M_g + BM - 1) / BM;
    const uint num_groups = (num_m_tiles + 1u) / 2u;  // pairs of 2

    // Scratch (8192 floats = 32KB):
    //   As0: 0    .. 1280 (BM*LDA)
    //   As1: 1280 .. 2560
    //   Bs:  2560 .. 4672 (BK*LDB = 2112)
    //   Cs:  0    .. 8192 (overlaps after MMA barrier)
    threadgroup float scratch[BM * BN];
    threadgroup float* As0 = scratch;
    threadgroup float* As1 = scratch + BM * LDA;
    threadgroup float* Bs  = scratch + 2u * BM * LDA;
    threadgroup float* Cs  = scratch;

    const uint sm = sgid / SIMDS_N;
    const uint sn = sgid % SIMDS_N;

    for (uint grp = tg_id; grp < num_groups; grp += n_tg) {
        const uint tile0 = grp * 2u;
        const uint c_row0_0 = tile0 * BM;
        const uint c_row0_1 = (tile0 + 1u) * BM;
        const bool has1 = (tile0 + 1u) < num_m_tiles;

        simdgroup_matrix<float, 8, 8> C0[MMA_M][MMA_N];
        simdgroup_matrix<float, 8, 8> C1[MMA_M][MMA_N];
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j) {
                C0[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
                C1[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
            }

        const uint num_k_tiles = (K_g + BK - 1) / BK;

        for (uint kt = 0; kt < num_k_tiles; ++kt) {
            const uint k0 = kt * BK;

            // Load A0, A1 (BM x BK each): 64*16 = 1024 → 1/thread each
            {
                const uint a_row = lid / BK;
                const uint a_col = lid & (BK - 1);
                const uint k_glob = k0 + a_col;
                float v0 = 0.0f, v1 = 0.0f;
                if (k_glob < K_g) {
                    const uint rr = k_glob / SC;
                    const uint rem = k_glob - rr * SC;
                    const uint ss = rem / C;
                    const uint c  = rem - ss * C;
                    {
                        const uint mg = c_row0_0 + a_row;
                        if (mg < M_g) {
                            const uint n_idx = mg / HW2;
                            const uint hw    = mg - n_idx * HW2;
                            const uint h2    = hw / W2;
                            const uint w2    = hw - h2 * W2;
                            v0 = x[((n_idx * H + h2 + rr) * W + w2 + ss) * C + c];
                        }
                    }
                    if (has1) {
                        const uint mg = c_row0_1 + a_row;
                        if (mg < M_g) {
                            const uint n_idx = mg / HW2;
                            const uint hw    = mg - n_idx * HW2;
                            const uint h2    = hw / W2;
                            const uint w2    = hw - h2 * W2;
                            v1 = x[((n_idx * H + h2 + rr) * W + w2 + ss) * C + c];
                        }
                    }
                }
                As0[a_row * LDA + a_col] = v0;
                As1[a_row * LDA + a_col] = v1;
            }

            // Load B (BK x BN): 16*128 = 2048 → 2/thread
            #pragma unroll
            for (uint p = 0; p < 2; ++p) {
                const uint t = lid + p * TG_THREADS;
                const uint b_col = t / BK;
                const uint b_row = t & (BK - 1);
                const uint kg = k0 + b_row;
                float v = 0.0f;
                if (kg < K_g) v = w[b_col * K_g + kg];
                Bs[b_row * LDB + b_col] = v;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // MMA: each kc step loads B once, both A blocks, MMA both tiles.
            #pragma unroll
            for (uint kc = 0; kc < BK; kc += 8u) {
                simdgroup_matrix<float, 8, 8> A0_blk[MMA_M], A1_blk[MMA_M], B_blk[MMA_N];
                #pragma unroll
                for (uint i = 0; i < MMA_M; ++i) {
                    simdgroup_load(A0_blk[i], &As0[(sm * SM + i * 8u) * LDA + kc], LDA);
                    simdgroup_load(A1_blk[i], &As1[(sm * SM + i * 8u) * LDA + kc], LDA);
                }
                #pragma unroll
                for (uint j = 0; j < MMA_N; ++j)
                    simdgroup_load(B_blk[j], &Bs[kc * LDB + sn * SN + j * 8u], LDB);
                #pragma unroll
                for (uint i = 0; i < MMA_M; ++i)
                    #pragma unroll
                    for (uint j = 0; j < MMA_N; ++j) {
                        simdgroup_multiply_accumulate(C0[i][j], A0_blk[i], B_blk[j], C0[i][j]);
                        simdgroup_multiply_accumulate(C1[i][j], A1_blk[i], B_blk[j], C1[i][j]);
                    }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Epilogue tile 0.
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_store(C0[i][j], &Cs[(sm * SM + i * 8u) * BN + sn * SN + j * 8u], BN);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint rs = 0; rs < 2; ++rs) {
            const uint r = (lid / 32u) + rs * 32u;
            const uint cq = lid & 31u;
            const uint c = cq * 4u;
            const uint m_g = c_row0_0 + r;
            if (m_g < M_g) {
                float4 v  = *reinterpret_cast<threadgroup float4*>(&Cs[r * BN + c]);
                float4 bv = *reinterpret_cast<device const float4*>(&b[c]);
                v += bv;
                v = fmax(v, float4(0.0f));
                const uint n_idx = m_g / HW2;
                const uint hw    = m_g - n_idx * HW2;
                const uint h2    = hw / W2;
                const uint w2    = hw - h2 * W2;
                *reinterpret_cast<device float4*>(&y[((n_idx * H2 + h2) * W2 + w2) * K_out + c]) = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Epilogue tile 1.
        if (has1) {
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                #pragma unroll
                for (uint j = 0; j < MMA_N; ++j)
                    simdgroup_store(C1[i][j], &Cs[(sm * SM + i * 8u) * BN + sn * SN + j * 8u], BN);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (uint rs = 0; rs < 2; ++rs) {
                const uint r = (lid / 32u) + rs * 32u;
                const uint cq = lid & 31u;
                const uint c = cq * 4u;
                const uint m_g = c_row0_1 + r;
                if (m_g < M_g) {
                    float4 v  = *reinterpret_cast<threadgroup float4*>(&Cs[r * BN + c]);
                    float4 bv = *reinterpret_cast<device const float4*>(&b[c]);
                    v += bv;
                    v = fmax(v, float4(0.0f));
                    const uint n_idx = m_g / HW2;
                    const uint hw    = m_g - n_idx * HW2;
                    const uint h2    = hw / W2;
                    const uint w2    = hw - h2 * W2;
                    *reinterpret_cast<device float4*>(&y[((n_idx * H2 + h2) * W2 + w2) * K_out + c]) = v;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
