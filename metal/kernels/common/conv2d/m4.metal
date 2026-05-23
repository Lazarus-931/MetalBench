// conv2d M4: implicit-im2col GEMM with simdgroup_matrix MMA, BK=32, float4 A/B loads.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM = 64, BN = 128, BK = 32;
constant constexpr uint SM = 16, SN = 16;
constant constexpr uint SIMDS_M = BM / SM;   // 4
constant constexpr uint SIMDS_N = BN / SN;   // 8
constant constexpr uint NUM_SG  = SIMDS_M * SIMDS_N;  // 32
constant constexpr uint MMA_M = SM / 8;      // 2
constant constexpr uint MMA_N = SN / 8;      // 2
constant constexpr uint TG_THREADS = 1024;
constant constexpr uint PAD = 4;
constant constexpr uint LDA = BK + PAD;      // 36
constant constexpr uint LDB = BN + PAD;      // 132

[[max_total_threads_per_threadgroup(1024)]]
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

    threadgroup float scratch[BM * BN];     // 8192 floats = 32 KB
    threadgroup float* As = scratch;
    threadgroup float* Bs = scratch + BM * LDA;     // offset 2304, length 32*132=4224
    threadgroup float* Cs = scratch;                // alias (used after K-loop)
    // Row metadata stashed after As+Bs (end=6528). 256 floats free → 4 BM-sized uint arrays.
    threadgroup uint* row_n = (threadgroup uint*)(scratch + 6528);          // [BM]
    threadgroup uint* row_h = (threadgroup uint*)(scratch + 6528 + BM);     // [BM]
    threadgroup uint* row_w = (threadgroup uint*)(scratch + 6528 + 2 * BM); // [BM]
    threadgroup uint* row_in_arr = (threadgroup uint*)(scratch + 6528 + 3 * BM); // [BM]

    const uint sm = sgid / SIMDS_N;          // 0..3
    const uint sn = sgid % SIMDS_N;          // 0..7

    // ---- Split 1024 threads into two halves so A-load and B-load issue in
    // parallel (different threads, no dependence). 512 do A, 512 do B.
    //
    // A-load (BM=64 × BK=32 = 2048 floats). 512 threads × 1 float4 = 2048. We
    // tile A as float4 along c (4 consecutive c values). 64 rows × 8 float4
    // = 512 jobs.
    const uint a_lid  = lid;                              // 0..511 active
    const uint a_row  = a_lid / 8u;                       // 0..63
    const uint a_col4 = a_lid & 7u;                       // 0..7 (float4 in c)
    const uint a_col0 = a_col4 * 4u;                      // 0..28

    // B-load (BK=32 × BN=128 = 4096 floats). 512 threads × 2 float4 = 4096.
    // Weights w[n_col, k] are K-contiguous → float4 along k.
    // 128 n_cols × 8 float4 (k_glob=k_base..k_base+3) = 1024 float4 jobs.
    // 512 threads × 2 float4 each.
    const uint b_lid  = lid - 512u;                        // 0..511 when lid>=512
    const uint b_col_a = b_lid / 8u;                       // 0..63   (1st N half)
    const uint b_k4    = b_lid & 7u;                       // 0..7
    const uint b_col_b = b_col_a + 64u;                    // 64..127 (2nd N half)

    // Persistent-threadgroup loop: each TG processes multiple BM-tiles.
    for (uint tile = tg_id; tile < num_m_tiles; tile += n_tg) {
        const uint c_row0 = tile * BM;

        simdgroup_matrix<float, 8, 8> C_acc[MMA_M][MMA_N];
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                C_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

        // Hoist per-row (n,h,w) into TG SMEM so the A-load thread (which owns
        // a_row = a_lid/8) can read its row coords without per-thread div.
        // Hoist per-row coords into shared scratch (aliased onto Cs region).
        if (lid < BM) {
            const uint m_g = c_row0 + lid;
            uint nv = 0, hv = 0, wv = 0, inb = 0;
            if (m_g < M_g) {
                nv = m_g / HW2;
                const uint rem = m_g - nv * HW2;
                hv = rem / W2;
                wv = rem - hv * W2;
                inb = 1;
            }
            row_n[lid] = nv; row_h[lid] = hv; row_w[lid] = wv;
            row_in_arr[lid] = inb;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint nv_a = row_n[a_row];
        const uint hv_a = row_h[a_row];
        const uint wv_a = row_w[a_row];
        const bool a_row_in = (row_in_arr[a_row] != 0);
        // Pre-compute the row-major x offset for (n, h0*stride, w0*stride, c=0)
        // so the K-loop only adds (rr*W + ss)*C + cbase.
        const uint x_row_base = ((nv_a * H + hv_a * stride_) * W + wv_a * stride_) * C;

        const uint num_k_tiles = (K_g + BK - 1) / BK;

        // K_g = R*S*C is divisible by BK (576 % 32 == 0 in registry), and each
        // BK-aligned k-tile sits inside a single (rr, ss) cell because BK | C
        // (32 | 64). Compute rr, ss, and c_base ONCE per k-tile (uniform across
        // threads) and only the per-thread c offset varies.
        for (uint kt = 0; kt < num_k_tiles; ++kt) {
            const uint k0 = kt * BK;
            const uint rr_t   = k0 / SC;
            const uint rem_t  = k0 - rr_t * SC;
            const uint ss_t   = rem_t / C;
            const uint cbase  = rem_t - ss_t * C;     // c for a_col=0
            // ---- A-load (lid < 512): 1 float4 per thread along c.
            // Each a_row's A-tile covers c=[cbase, cbase+BK). Within a tile,
            // 4 consecutive a_col → 4 consecutive c (no boundary crossing).
            if (lid < 512u) {
                float4 v = float4(0.0f);
                if (a_row_in) {
                    // (h0+rr)*W + (w0+ss) - (h0*W + w0) = rr*W + ss; multiplied
                    // by C and added to cbase gives the per-k offset.
                    const uint k_off = (rr_t * W + ss_t) * C + cbase + a_col0;
                    v = *reinterpret_cast<const device float4*>(
                            &x[x_row_base + k_off]);
                }
                *reinterpret_cast<threadgroup float4*>(&As[a_row * LDA + a_col0]) = v;
            } else {
            // ---- B-load (lid >= 512): 2 float4s per thread, K-contiguous.
                const uint k_base = k0 + b_k4 * 4u;
                const uint k_row  = b_k4 * 4u;
                // K-major Bs layout (k=row, n=col). Float4 from w (K-contig)
                // is scattered to 4 distinct k-rows. Two N halves per thread.
                {
                    float4 va = *reinterpret_cast<const device float4*>(
                            &w[b_col_a * K_g + k_base]);
                    Bs[(k_row + 0u) * LDB + b_col_a] = va.x;
                    Bs[(k_row + 1u) * LDB + b_col_a] = va.y;
                    Bs[(k_row + 2u) * LDB + b_col_a] = va.z;
                    Bs[(k_row + 3u) * LDB + b_col_a] = va.w;
                }
                {
                    float4 vb = *reinterpret_cast<const device float4*>(
                            &w[b_col_b * K_g + k_base]);
                    Bs[(k_row + 0u) * LDB + b_col_b] = vb.x;
                    Bs[(k_row + 1u) * LDB + b_col_b] = vb.y;
                    Bs[(k_row + 2u) * LDB + b_col_b] = vb.z;
                    Bs[(k_row + 3u) * LDB + b_col_b] = vb.w;
                }
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

        // Stage simdgroup tiles to SMEM Cs, then float4 scatter to device y.
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
                const uint m_g = c_row0 + r;
                if (m_g < M_g) {
                    const uint nv = m_g / HW2;
                    const uint rem = m_g - nv * HW2;
                    const uint hv = rem / W2;
                    const uint wv = rem - hv * W2;
                    float4 v = *reinterpret_cast<threadgroup float4*>(&Cs[r * BN + cc]);
                    *reinterpret_cast<device float4*>(&y[((nv * H2 + hv) * W2 + wv) * K_out + cc]) = v;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
