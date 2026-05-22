// conv2d implicit im2col GEMM with simdgroup_matrix MMA, M4-tuned.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM = 64, BN = 128, BK = 16;
constant constexpr uint SM = 16, SN = 16;
constant constexpr uint SIMDS_M = BM / SM;   // 4
constant constexpr uint SIMDS_N = BN / SN;   // 8
constant constexpr uint NUM_SG  = SIMDS_M * SIMDS_N;  // 32
constant constexpr uint MMA_M = SM / 8;      // 2
constant constexpr uint MMA_N = SN / 8;      // 2
constant constexpr uint TG_THREADS = 1024;
constant constexpr uint PAD = 4;
constant constexpr uint LDA = BK + PAD;      // 20
constant constexpr uint LDB = BN + PAD;      // 132

kernel void conv2d_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  w       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   N_      [[buffer(3)]],
    constant     uint&   C       [[buffer(4)]],
    constant     uint&   H       [[buffer(5)]],
    constant     uint&   W       [[buffer(6)]],
    constant     uint&   K_out   [[buffer(7)]],
    constant     uint&   R       [[buffer(8)]],
    constant     uint&   S       [[buffer(9)]],
    constant     uint&   stride_ [[buffer(10)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint sgid    [[simdgroup_index_in_threadgroup]],
    uint lid     [[thread_index_in_threadgroup]],
    uint n_tg    [[threadgroups_per_grid]])
{
    const uint H2 = (H - R) / stride_ + 1;
    const uint W2 = (W - S) / stride_ + 1;
    const uint M_g = N_ * H2 * W2;
    const uint K_g = R * S * C;
    const uint SC  = S * C;
    const uint num_m_tiles = (M_g + BM - 1) / BM;
    const uint total_tiles = num_m_tiles;

    threadgroup float scratch[BM * BN];
    threadgroup float* As = scratch;
    threadgroup float* Bs = scratch + BM * LDA;     // 1280 offset
    threadgroup float* Cs = scratch;                // alias (used after K-loop)

    const uint sm = sgid / SIMDS_N;          // 0..3
    const uint sn = sgid % SIMDS_N;          // 0..7

    for (uint tile = tg_id; tile < total_tiles; tile += n_tg) {
        const uint c_row0 = tile * BM;

        simdgroup_matrix<float, 8, 8> C_acc[MMA_M][MMA_N];
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                C_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

        const uint num_k_tiles = (K_g + BK - 1) / BK;

        for (uint kt = 0; kt < num_k_tiles; ++kt) {
            const uint k0 = kt * BK;

            {
                const uint a_row = lid / BK;
                const uint a_col = lid % BK;
                const uint m_g = c_row0 + a_row;
                const uint k_glob = k0 + a_col;
                float v = 0.0f;
                if (m_g < M_g && k_glob < K_g) {
                    const uint rr = k_glob / SC;
                    const uint rem = k_glob - rr * SC;
                    const uint ss = rem / C;
                    const uint c  = rem - ss * C;
                    const uint n_idx = m_g / (H2 * W2);
                    const uint hw = m_g - n_idx * (H2 * W2);
                    const uint h2 = hw / W2;
                    const uint w2 = hw - h2 * W2;
                    const uint hi = h2 * stride_ + rr;
                    const uint wi = w2 * stride_ + ss;
                    v = x[((n_idx * H + hi) * W + wi) * C + c];
                }
                As[a_row * LDA + a_col] = v;
            }
            #pragma unroll
            for (uint p = 0; p < 2; ++p) {
                const uint t = lid + p * TG_THREADS;
                const uint b_col = t / BK;
                const uint b_row = t % BK;
                const uint k_glob = k0 + b_row;
                const uint n_col = b_col;
                float v = 0.0f;
                if (k_glob < K_g && n_col < K_out) {
                    v = w[n_col * K_g + k_glob];
                }
                Bs[b_row * LDB + b_col] = v;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll
            for (uint kc = 0; kc < BK; kc += 8u) {
                simdgroup_matrix<float, 8, 8> A_blk[MMA_M], B_blk[MMA_N];
                #pragma unroll
                for (uint i = 0; i < MMA_M; ++i)
                    simdgroup_load(A_blk[i], &As[(sm * SM + i * 8u) * LDA + kc], LDA);
                #pragma unroll
                for (uint j = 0; j < MMA_N; ++j)
                    simdgroup_load(B_blk[j], &Bs[kc * LDB + sn * SN + j * 8u], LDB);
                #pragma unroll
                for (uint i = 0; i < MMA_M; ++i)
                    #pragma unroll
                    for (uint j = 0; j < MMA_N; ++j)
                        simdgroup_multiply_accumulate(C_acc[i][j], A_blk[i], B_blk[j], C_acc[i][j]);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_store(C_acc[i][j], &Cs[(sm * SM + i * 8u) * BN + sn * SN + j * 8u], BN);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            #pragma unroll
            for (uint rs = 0; rs < 2; ++rs) {
                const uint r = (lid / 32u) + rs * 32u;
                const uint cq = lid % 32u;
                const uint c = cq * 4u;
                const uint m_g = c_row0 + r;
                if (m_g < M_g) {
                    const uint n_idx = m_g / (H2 * W2);
                    const uint hw = m_g - n_idx * (H2 * W2);
                    const uint h2 = hw / W2;
                    const uint w2 = hw - h2 * W2;
                    float4 v = *reinterpret_cast<threadgroup float4*>(&Cs[r * BN + c]);
                    *reinterpret_cast<device float4*>(&y[((n_idx * H2 + h2) * W2 + w2) * K_out + c]) = v;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
