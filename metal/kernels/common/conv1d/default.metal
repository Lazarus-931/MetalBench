// conv1d implicit im2col GEMM with simdgroup_matrix MMA.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM = 64, BN = 64, BK = 16;
constant constexpr uint SM = 16, SN = 8;
constant constexpr uint SIMDS_N = BN / SN;   // 8
constant constexpr uint MMA_M = SM / 8;      // 2
constant constexpr uint MMA_N = SN / 8;      // 1
constant constexpr uint PAD = 4;
constant constexpr uint LDA = BK + PAD;      // 20
constant constexpr uint LDB = BN + PAD;      // 68

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

    threadgroup float As[BM * LDA];
    threadgroup float Bs[BK * LDB];
    threadgroup float Cs[BM * BN];

    const uint sm = sgid / SIMDS_N, sn = sgid % SIMDS_N;

    for (uint tile = tg_id; tile < total_tiles; tile += n_tg) {
        const uint mtile = tile / num_n_tiles;
        const uint ntile = tile % num_n_tiles;
        const uint c_row0 = mtile * BM;
        const uint c_col0 = ntile * BN;

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
                const uint a_row = lid / 16u;
                const uint a_col = lid % 16u;
                const uint m_g = c_row0 + a_row;
                float v = 0.0f;
                if (m_g < M_g) {
                    const uint n_idx = m_g / L2;
                    const uint l2_idx = m_g - n_idx * L2;
                    const uint li = l2_idx * stride_ + rr;
                    v = x[(n_idx * L + li) * C + c_base + a_col];
                }
                As[a_row * LDA + a_col] = v;
            }
            {
                const uint b_col = lid / BK;
                const uint b_row = lid % BK;
                const uint k_glob = k0 + b_row;
                const uint n_col = c_col0 + b_col;
                Bs[b_row * LDB + b_col] = w[n_col * K_g + k_glob];
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
            const uint idx = lid;
            const uint r = (idx * 4u) / BN;
            const uint c = (idx * 4u) % BN;
            const uint m_g = c_row0 + r;
            if (m_g < M_g) {
                const uint n_idx = m_g / L2;
                const uint l2_idx = m_g - n_idx * L2;
                float4 v = *reinterpret_cast<threadgroup float4*>(&Cs[r * BN + c]);
                *reinterpret_cast<device float4*>(&y[(n_idx * L2 + l2_idx) * K_out + c_col0 + c]) = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
