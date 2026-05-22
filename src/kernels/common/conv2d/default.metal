// conv2d implicit-im2col GEMM with simdgroup_matrix MMA.
// M4 wins over the previous (M4) kernel:
//   - K-tile double-buffering: load next tile (A and B) while compute
//     proceeds on the current. BK=16 chosen so As + Bs (×2) + the aliased
//     Cs region fits in the 32 KB threadgroup memory.
//   - Hoist per-row (n_idx, h2, w2) im2col coords out of the K-loop — these
//     are constant across all 36 K-iterations and the previous kernel
//     recomputed them every step.
// SMEM: As[2][64×20] = 2560 + Bs[2][16×132] = 4224 = 6784 floats during the
// K-loop; Cs[64×128] = 8192 floats (aliased over the same scratch) during the
// epilogue. Max = 8192 floats = 32 KB (M4 limit).
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
constant constexpr uint AS_STRIDE = BM * LDA;   // 1280
constant constexpr uint BS_STRIDE = BK * LDB;   // 2112

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
    const uint HW2 = H2 * W2;
    const uint num_m_tiles = (M_g + BM - 1) / BM;

    threadgroup float scratch[BM * BN];
    threadgroup float* Asb[2] = { scratch, scratch + AS_STRIDE };
    threadgroup float* Bsb[2] = { scratch + 2u * AS_STRIDE,
                                  scratch + 2u * AS_STRIDE + BS_STRIDE };
    threadgroup float* Cs = scratch;

    const uint sm = sgid / SIMDS_N;          // 0..3
    const uint sn = sgid % SIMDS_N;          // 0..7

    // ---- Thread-private indexing for A / B loads.
    const uint a_row = lid / BK;             // 0..63
    const uint a_col = lid - a_row * BK;     // 0..15

    // Persistent-threadgroup loop: each TG processes multiple BM-tiles.
    for (uint tile = tg_id; tile < num_m_tiles; tile += n_tg) {
        const uint c_row0 = tile * BM;

        simdgroup_matrix<float, 8, 8> C_acc[MMA_M][MMA_N];
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                C_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

        // Hoist per-row im2col coords (constant across the K-loop).
        const uint m_g = c_row0 + a_row;
        const bool row_in = (m_g < M_g);
        uint nv = 0, hv = 0, wv = 0;
        if (row_in) {
            nv = m_g / HW2;
            const uint rem = m_g - nv * HW2;
            hv = rem / W2;
            wv = rem - hv * W2;
        }

        const uint num_k_tiles = (K_g + BK - 1) / BK;

        // ---- Prologue: load K-tile 0 into buffer 0.
        {
            const uint k0 = 0;
            const uint k_glob = k0 + a_col;
            float va = 0.0f;
            if (row_in && k_glob < K_g) {
                const uint rr  = k_glob / SC;
                const uint rem = k_glob - rr * SC;
                const uint ss  = rem / C;
                const uint cc  = rem - ss * C;
                const uint hi  = hv * stride_ + rr;
                const uint wi  = wv * stride_ + ss;
                va = x[((nv * H + hi) * W + wi) * C + cc];
            }
            Asb[0][a_row * LDA + a_col] = va;
            #pragma unroll
            for (uint pp = 0; pp < 2; ++pp) {
                const uint t = lid + pp * TG_THREADS;
                const uint b_col = t / BK;
                const uint b_row = t - b_col * BK;
                const uint k_g_b = k0 + b_row;
                float vb = 0.0f;
                if (k_g_b < K_g && b_col < K_out) {
                    vb = w[b_col * K_g + k_g_b];
                }
                Bsb[0][b_row * LDB + b_col] = vb;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint buf = 0;
        for (uint kt = 0; kt < num_k_tiles - 1; ++kt) {
            const uint next = 1u - buf;
            const uint k0_nxt = (kt + 1u) * BK;

            // ---- Prefetch next K-tile into the other buffer.
            const uint k_glob = k0_nxt + a_col;
            float va = 0.0f;
            if (row_in && k_glob < K_g) {
                const uint rr  = k_glob / SC;
                const uint rem = k_glob - rr * SC;
                const uint ss  = rem / C;
                const uint cc  = rem - ss * C;
                const uint hi  = hv * stride_ + rr;
                const uint wi  = wv * stride_ + ss;
                va = x[((nv * H + hi) * W + wi) * C + cc];
            }
            Asb[next][a_row * LDA + a_col] = va;

            #pragma unroll
            for (uint pp = 0; pp < 2; ++pp) {
                const uint t = lid + pp * TG_THREADS;
                const uint b_col = t / BK;
                const uint b_row = t - b_col * BK;
                const uint k_g_b = k0_nxt + b_row;
                float vb = 0.0f;
                if (k_g_b < K_g && b_col < K_out) {
                    vb = w[b_col * K_g + k_g_b];
                }
                Bsb[next][b_row * LDB + b_col] = vb;
            }

            // ---- Compute on the current buffer.
            #pragma unroll
            for (uint kc = 0; kc < BK; kc += 8u) {
                simdgroup_matrix<float, 8, 8> A_blk[MMA_M], B_blk[MMA_N];
                #pragma unroll
                for (uint i = 0; i < MMA_M; ++i)
                    simdgroup_load(A_blk[i], &Asb[buf][(sm * SM + i * 8u) * LDA + kc], LDA);
                #pragma unroll
                for (uint j = 0; j < MMA_N; ++j)
                    simdgroup_load(B_blk[j], &Bsb[buf][kc * LDB + sn * SN + j * 8u], LDB);
                #pragma unroll
                for (uint i = 0; i < MMA_M; ++i)
                    #pragma unroll
                    for (uint j = 0; j < MMA_N; ++j)
                        simdgroup_multiply_accumulate(C_acc[i][j], A_blk[i], B_blk[j], C_acc[i][j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf = next;
        }

        // ---- Tail: compute on the last loaded buffer.
        #pragma unroll
        for (uint kc = 0; kc < BK; kc += 8u) {
            simdgroup_matrix<float, 8, 8> A_blk[MMA_M], B_blk[MMA_N];
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                simdgroup_load(A_blk[i], &Asb[buf][(sm * SM + i * 8u) * LDA + kc], LDA);
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_load(B_blk[j], &Bsb[buf][kc * LDB + sn * SN + j * 8u], LDB);
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                #pragma unroll
                for (uint j = 0; j < MMA_N; ++j)
                    simdgroup_multiply_accumulate(C_acc[i][j], A_blk[i], B_blk[j], C_acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stage simdgroup tiles to SMEM Cs (aliased), then float4 scatter to y.
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
                const uint cc = cq * 4u;
                const uint m_g_e = c_row0 + r;
                if (m_g_e < M_g) {
                    const uint nv_e = m_g_e / HW2;
                    const uint rem = m_g_e - nv_e * HW2;
                    const uint hv_e = rem / W2;
                    const uint wv_e = rem - hv_e * W2;
                    float4 v = *reinterpret_cast<threadgroup float4*>(&Cs[r * BN + cc]);
                    *reinterpret_cast<device float4*>(&y[((nv_e * H2 + hv_e) * W2 + wv_e) * K_out + cc]) = v;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
