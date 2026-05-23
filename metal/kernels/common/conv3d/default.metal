// conv3d implicit im2col GEMM with simdgroup_matrix MMA.
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

kernel void conv3d_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  w       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   N_      [[buffer(3)]],
    constant     uint&   C       [[buffer(4)]],
    constant     uint&   D       [[buffer(5)]],
    constant     uint&   H       [[buffer(6)]],
    constant     uint&   W       [[buffer(7)]],
    constant     uint&   K_out   [[buffer(8)]],
    constant     uint&   R       [[buffer(9)]],
    constant     uint&   stride_ [[buffer(10)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint sgid    [[simdgroup_index_in_threadgroup]],
    uint lid     [[thread_index_in_threadgroup]],
    uint n_tg    [[threadgroups_per_grid]])
{
    const uint D2 = (D - R) / stride_ + 1;
    const uint H2 = (H - R) / stride_ + 1;
    const uint W2 = (W - R) / stride_ + 1;
    const uint HW2 = H2 * W2;
    const uint DHW2 = D2 * HW2;
    const uint M_g = N_ * DHW2;
    const uint RC = R * C;
    const uint RRC = R * RC;
    const uint K_g = R * RRC;
    const uint num_m_tiles = (M_g + BM - 1) / BM;
    const uint num_n_tiles = (K_out + BN - 1) / BN;
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

        const uint num_k_tiles = (K_g + BK - 1) / BK;

        for (uint kt = 0; kt < num_k_tiles; ++kt) {
            const uint k0 = kt * BK;

            {
                const uint a_row = lid / 16u;
                const uint a_col = lid % 16u;
                const uint k_glob = k0 + a_col;
                const uint rd = k_glob / RRC;
                uint rem = k_glob - rd * RRC;
                const uint rh = rem / RC;
                rem -= rh * RC;
                const uint rw = rem / C;
                const uint c  = rem - rw * C;

                const uint m_g = c_row0 + a_row;
                float v = 0.0f;
                if (m_g < M_g && k_glob < K_g) {
                    const uint n_idx = m_g / DHW2;
                    uint q = m_g - n_idx * DHW2;
                    const uint d_idx = q / HW2;
                    q -= d_idx * HW2;
                    const uint h_idx = q / W2;
                    const uint w_idx = q - h_idx * W2;
                    const uint di = d_idx * stride_ + rd;
                    const uint hi = h_idx * stride_ + rh;
                    const uint wi = w_idx * stride_ + rw;
                    v = x[(((n_idx * D + di) * H + hi) * W + wi) * C + c];
                }
                As[a_row * LDA + a_col] = v;
            }
            {
                const uint b_col = lid / BK;
                const uint b_row = lid % BK;
                const uint k_glob = k0 + b_row;
                const uint n_col = c_col0 + b_col;
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
            const uint idx = lid;
            const uint r = (idx * 4u) / BN;
            const uint c = (idx * 4u) % BN;
            const uint m_g = c_row0 + r;
            if (m_g < M_g) {
                const uint n_idx = m_g / DHW2;
                uint q = m_g - n_idx * DHW2;
                const uint d_idx = q / HW2;
                q -= d_idx * HW2;
                const uint h_idx = q / W2;
                const uint w_idx = q - h_idx * W2;
                float4 v = *reinterpret_cast<threadgroup float4*>(&Cs[r * BN + c]);
                const uint nc = c_col0 + c;
                const uint y_base = (((n_idx * D2 + d_idx) * H2 + h_idx) * W2 + w_idx) * K_out + nc;
                if (nc + 3 < K_out) {
                    *reinterpret_cast<device float4*>(&y[y_base]) = v;
                } else {
                    float vs[4] = {v.x, v.y, v.z, v.w};
                    for (uint kk = 0; kk < 4u; ++kk) {
                        if (nc + kk < K_out)
                            y[y_base + kk] = vs[kk];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
