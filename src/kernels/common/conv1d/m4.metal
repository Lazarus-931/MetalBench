// conv1d implicit im2col GEMM with simdgroup_matrix MMA (M4-tuned v2).
//
// Logical GEMM: M_g=2032, K_g=192, N_g=128.
// Tiling: BM=64, BN=64, BK=32 → 6 K-tiles; 1024 threads/TG = 32 simdgroups.
// SM=16, SN=8 → MMA_M=2, MMA_N=1, register C accumulation.
//
// Notes for M4 (10 GPU cores):
//   * Workload is tiny — total FLOPs ≈ 100M. With registry-fixed
//     threadgroup=(1024,1,1) we get only 64 TGs (grid 65536/1024).
//     Each TG = 32 simdgroups; M4 cores fit ~1-2 such heavy TGs each →
//     ~10-20 TGs concurrent → kernel is launch/occupancy-bound at this size.
//   * Tile-shrink for occupancy is blocked by registry threadgroup=1024
//     (the SM allocates per-TG, so reducing per-tile threads still consumes
//     the same TG slot). Re-tested register-pipelined prefetch (v3) – it
//     regressed (48 ms). Sticking with v2 PAD=0 + full-lane loads.
//
// Wins:
//   * PAD=0 saves SMEM (As: 64*32, Bs: 32*64).
//   * Wider A/B loads using ALL 1024 threads (no idle lanes).
//   * Direct simdgroup_store to global for full m-tiles.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM = 64, BN = 64, BK = 32;
constant constexpr uint SM = 16, SN = 8;
constant constexpr uint SIMDS_N = BN / SN;   // 8
constant constexpr uint MMA_M = SM / 8;      // 2
constant constexpr uint MMA_N = SN / 8;      // 1
constant constexpr uint LDA = BK;            // 32, PAD=0
constant constexpr uint LDB = BN;            // 64, PAD=0

kernel void conv1d_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  w       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   N_      [[buffer(3)]],
    constant     uint&   C       [[buffer(4)]],
    constant     uint&   L       [[buffer(5)]],
    constant     uint&   K_out   [[buffer(6)]],
    constant     uint&   R       [[buffer(7)]],
    constant     uint&   stride_ [[buffer(8)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint sgid    [[simdgroup_index_in_threadgroup]],
    uint lid     [[thread_index_in_threadgroup]],
    uint n_tg    [[threadgroups_per_grid]])
{
    const uint L2 = (L - R) / stride_ + 1;
    const uint M_g = N_ * L2;
    const uint K_g = R * C;
    const uint num_m_tiles = (M_g + BM - 1) / BM;
    const uint num_n_tiles = K_out / BN;
    const uint total_tiles = num_m_tiles * num_n_tiles;

    threadgroup float Smem[(BM * BN > BM * LDA + BK * LDB) ? BM * BN : BM * LDA + BK * LDB];
    threadgroup float* As = Smem;
    threadgroup float* Bs = Smem + BM * LDA;
    threadgroup float* Cs = Smem;

    const uint sm = sgid / SIMDS_N;
    const uint sn = sgid % SIMDS_N;

    for (uint tile = tg_id; tile < total_tiles; tile += n_tg) {
        const uint mtile  = tile / num_n_tiles;
        const uint ntile  = tile - mtile * num_n_tiles;
        const uint c_row0 = mtile * BM;
        const uint c_col0 = ntile * BN;
        const bool partial_m = (c_row0 + BM) > M_g;

        simdgroup_matrix<float, 8, 8> C_acc[MMA_M][MMA_N];
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                C_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

        const uint num_k_tiles = K_g / BK;

        for (uint kt = 0; kt < num_k_tiles; ++kt) {
            const uint k0 = kt * BK;
            const uint rr = k0 / C;
            const uint c_base = k0 - rr * C;

            {
                const uint a_row = lid >> 4;
                const uint a_c2  = lid & 15u;
                const uint col = a_c2 * 2u;
                const uint m_g = c_row0 + a_row;
                float2 v = float2(0.0f);
                if (m_g < M_g) {
                    const uint n_idx  = m_g / L2;
                    const uint l2_idx = m_g - n_idx * L2;
                    const uint li     = l2_idx * stride_ + rr;
                    v = *reinterpret_cast<const device float2*>(
                        &x[(n_idx * L + li) * C + c_base + col]);
                }
                *reinterpret_cast<threadgroup float2*>(&As[a_row * LDA + col]) = v;
            }

            {
                const uint b_col   = lid >> 4;
                const uint b_row0  = (lid & 15u) * 2u;
                const uint k_glob  = k0 + b_row0;
                const uint n_col   = c_col0 + b_col;
                const device float2* wp = reinterpret_cast<const device float2*>(
                    &w[n_col * K_g + k_glob]);
                float2 v = *wp;
                Bs[(b_row0 + 0u) * LDB + b_col] = v.x;
                Bs[(b_row0 + 1u) * LDB + b_col] = v.y;
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

        if (!partial_m) {
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i) {
                #pragma unroll
                for (uint j = 0; j < MMA_N; ++j) {
                    const uint r = c_row0 + sm * SM + i * 8u;
                    const uint c = c_col0 + sn * SN + j * 8u;
                    simdgroup_store(C_acc[i][j], &y[r * K_out + c], K_out);
                }
            }
        } else {
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i) {
                #pragma unroll
                for (uint j = 0; j < MMA_N; ++j) {
                    simdgroup_store(C_acc[i][j],
                                    &Cs[(sm * SM + i * 8u) * BN + sn * SN + j * 8u], BN);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint idx = lid;
            const uint r   = (idx * 4u) / BN;
            const uint c   = (idx * 4u) - r * BN;
            const uint m_g = c_row0 + r;
            if (m_g < M_g) {
                float4 v = *reinterpret_cast<threadgroup float4*>(&Cs[r * BN + c]);
                *reinterpret_cast<device float4*>(&y[m_g * K_out + c_col0 + c]) = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
