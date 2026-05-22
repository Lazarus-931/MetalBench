// Fused conv3d -> softmax(C) -> maxpool2x2x2 -> maxpool2x2x2.
// One threadgroup per output block (n, d4, h4, w4) in (4,7,7,7) = 1372 blocks.
//
// GEMM: M=64 (voxels in 4x4x4), N=64 (K_out channels), K=R^3*C = 864.
// Tile BM=64, BN=64, BK=32. 8 simdgroups (SIMDS_M=2, SIMDS_N=4); 256 threads/TG.
// Per simd: SM=32, SN=16 → MMA_M=4, MMA_N=2 = 8 MMAs × 4 inner kc steps = 32 MMAs / K-tile.
//
// Key optimizations vs previous version:
//   * Hoist 3D index decomposition out of inner load loop — since BK == C == 32 and
//     k0 % 32 == 0, (rd, rh, rw) are constant per kt iter, and c == a_col.
//   * float4 vector loads from device → threadgroup for both x and w.
//   * Simdgroup-shuffle reduction for the 4-way partial-max merge after softmax.
//   * Reciprocal of row-sum stored once so per-cell uses multiply not divide.
//   * fast::exp for the softmax.
//   * Reduced threadgroup memory by overlapping GEMM scratch (As/Bs) with post-GEMM Cs.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM = 64, BN = 64, BK = 32;
constant constexpr uint SIMDS_M = 2, SIMDS_N = 4;
constant constexpr uint SM = BM / SIMDS_M;           // 32
constant constexpr uint SN = BN / SIMDS_N;           // 16
constant constexpr uint MMA_M = SM / 8;              // 4
constant constexpr uint MMA_N = SN / 8;              // 2
constant constexpr uint PAD = 4;
constant constexpr uint LDA = BK + PAD;              // 36
constant constexpr uint LDB = BN + PAD;              // 68

kernel void conv3d_softmax_pool_f32(
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
    uint tg_id   [[threadgroup_position_in_grid]],
    uint sgid    [[simdgroup_index_in_threadgroup]],
    uint lid     [[thread_index_in_threadgroup]],
    uint n_tg    [[threadgroups_per_grid]])
{
    const uint D_out = 7, H_out = 7, W_out = 7;
    const uint blocks_per_n = D_out * H_out * W_out;
    const uint total_blocks = N_ * blocks_per_n;
    const uint RC = R * C;
    const uint RRC = R * RC;
    const uint K_g = R * RRC;  // 864

    // Memory layout (overlap GEMM scratch with softmax/maxpool scratch).
    // GEMM phase:    As [BM*LDA] then Bs [BK*LDB] in the same buffer.
    // Post-GEMM:     Cs [BM*BN] reuses the start of the buffer.
    // Max needed:    max(BM*LDA + BK*LDB, BM*BN) = max(2304+2176, 4096) = 4480 floats (~18KB).
    threadgroup float scratch[BM * LDA + BK * LDB];   // 4480 floats
    threadgroup float* As = scratch;
    threadgroup float* Bs = scratch + BM * LDA;
    threadgroup float* Cs = scratch;
    threadgroup float row_m[BM];
    threadgroup float row_Z[BM];   // stores 1 / Z so per-cell softmax uses multiply

    const uint sm = sgid / SIMDS_N;
    const uint sn = sgid % SIMDS_N;

    for (uint blk = tg_id; blk < total_blocks; blk += n_tg) {
        uint q = blk;
        const uint w4 = q % W_out; q /= W_out;
        const uint h4 = q % H_out; q /= H_out;
        const uint d4 = q % D_out;
        const uint n  = q / D_out;
        const uint d0 = d4 * 4;
        const uint h0 = h4 * 4;
        const uint w0 = w4 * 4;

        simdgroup_matrix<float, 8, 8> C_acc[MMA_M][MMA_N];
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                C_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

        const uint num_k_tiles = (K_g + BK - 1) / BK;   // 27 with BK=32

        for (uint kt = 0; kt < num_k_tiles; ++kt) {
            const uint k0 = kt * BK;
            // Since BK == C == 32 and k0 % 32 == 0, the convolution offsets (rd, rh, rw)
            // are constant across the entire K-tile and c == a_col.
            const uint kt_idx = k0 >> 5;
            const uint rw = kt_idx % R;
            const uint rh = (kt_idx / R) % R;
            const uint rd = kt_idx / (R * R);

            // Load A: BM*BK = 64*32 = 2048 floats via 512 float4 reads (256 thr × 2 iters).
            // pos = it*256 + lid. a_row = pos / 8 ∈ [0,64). chunk = pos % 8 ∈ [0,8). a_col = chunk*4.
            {
                const device float4* x4 = reinterpret_cast<const device float4*>(x);
                #pragma unroll
                for (uint it = 0; it < 2u; ++it) {
                    const uint pos = it * 256u + lid;
                    const uint a_row = pos >> 3;
                    const uint chunk = pos & 7u;
                    const uint a_col = chunk * 4u;
                    const uint dd = a_row >> 4;
                    const uint hh = (a_row >> 2) & 3u;
                    const uint ww = a_row & 3u;
                    const uint di = d0 + dd + rd;
                    const uint hi = h0 + hh + rh;
                    const uint wi = w0 + ww + rw;
                    const uint addr = (((n * D + di) * H + hi) * W + wi) * C + a_col;
                    float4 v = x4[addr >> 2];
                    As[a_row * LDA + a_col + 0u] = v.x;
                    As[a_row * LDA + a_col + 1u] = v.y;
                    As[a_row * LDA + a_col + 2u] = v.z;
                    As[a_row * LDA + a_col + 3u] = v.w;
                }
            }
            // Load B: BK*BN = 32*64 = 2048 floats via 512 float4 reads (256 thr × 2 iters).
            // pos = it*256 + lid. b_col = pos / 8 ∈ [0,64). chunk = pos % 8 ∈ [0,8). k_glob = k0 + chunk*4.
            {
                const device float4* w4 = reinterpret_cast<const device float4*>(w);
                #pragma unroll
                for (uint it = 0; it < 2u; ++it) {
                    const uint pos = it * 256u + lid;
                    const uint b_col = pos >> 3;
                    const uint chunk = pos & 7u;
                    const uint k_glob = k0 + chunk * 4u;
                    // K_g = 27*BK exactly; BN = K_out = 64 → no bounds needed.
                    float4 v = w4[(b_col * K_g + k_glob) >> 2];
                    Bs[(chunk * 4u + 0u) * LDB + b_col] = v.x;
                    Bs[(chunk * 4u + 1u) * LDB + b_col] = v.y;
                    Bs[(chunk * 4u + 2u) * LDB + b_col] = v.z;
                    Bs[(chunk * 4u + 3u) * LDB + b_col] = v.w;
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

        // Store accumulator to Cs (row=voxel, col=channel).
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_store(C_acc[i][j], &Cs[(sm * SM + i * 8u) * BN + sn * SN + j * 8u], BN);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-row softmax stats over the 64 channels.
        // 256 threads, 64 rows → 4 thr/row. We use: each simd handles 8 rows, lane reads 2 channels.
        // simd_max / simd_sum reduce across the 32 lanes of a simdgroup ⇒ 64 channels total.
        {
            const uint lane = lid & 31u;
            #pragma unroll
            for (uint rr = 0; rr < 8u; ++rr) {
                const uint row = sgid * 8u + rr;
                float c0 = Cs[row * BN + lane * 2u + 0u];
                float c1 = Cs[row * BN + lane * 2u + 1u];
                float lm = max(c0, c1);
                float m = simd_max(lm);
                float e0 = fast::exp(c0 - m);
                float e1 = fast::exp(c1 - m);
                float ls = e0 + e1;
                float Z = simd_sum(ls);
                if (lane == 0) {
                    row_m[row] = m;
                    row_Z[row] = 1.0f / Z;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Softmax + reduce-max across 64 voxels for each of 64 channels.
        // 256 threads, 64 channels → 4 thr/ch. Each thread handles 16 rows then
        // reduces with two simd_shuffle_xor calls (lanes 4k..4k+3 hold partials for one channel).
        {
            const uint ch = lid >> 2;
            const uint slot = lid & 3u;
            float best = -INFINITY;
            #pragma unroll
            for (uint r = 0; r < 16u; ++r) {
                uint row = slot * 16u + r;
                float cv = Cs[row * BN + ch];
                float v = fast::exp(cv - row_m[row]) * row_Z[row];
                best = max(best, v);
            }
            best = max(best, simd_shuffle_xor(best, 1u));
            best = max(best, simd_shuffle_xor(best, 2u));
            if (slot == 0u) {
                const uint y_idx = (((n * D_out + d4) * H_out + h4) * W_out + w4) * K_out + ch;
                y[y_idx] = best;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
