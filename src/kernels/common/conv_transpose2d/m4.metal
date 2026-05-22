// conv_transpose2d on M4: implicit-im2col GEMM with simdgroup_matrix MMA.
// Shapes: NHWC, x(N=8, H=32, W=32, C_in=64), w(C_out=128, R=3, S=3, C_in=64), stride=2.
// Output: (8, 65, 65, 128). M=N*H_out*W_out=33800, N=128, K=R*S*C_in=576.
//
// Strategy:
//   - TG output tile (BM=128 spatial rows) x (BN=128 = full OUT_K).
//   - 32 simdgroups (1024 threads); 16 row blocks x 2 col groups of 64 cols.
//     Each SG owns 8 8x8 tiles spanning the full BN.
//   - Outer loop over (rr,ss) (9 iters) with BK=32 sub-iter (2 per rr,ss).
//   - For each step: stage Atile (128x32 im2col gather, zero-padded for
//     stride-parity / OOB) and Btile (32x128 weight slab).
//   - Grid-agnostic striding over m-tiles using tg_id / n_tg.
//   - Final store: fast simdgroup_store path when all 8 rows of the SG block
//     are in bounds, else scratch + scalar fallback.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint C_IN  = 64;
constant constexpr uint OUT_K = 128;
constant constexpr uint R_K   = 3;
constant constexpr uint S_K   = 3;
constant constexpr uint KD    = R_K * S_K * C_IN; // 576

constant constexpr uint BM = 128;
constant constexpr uint BN = 128;          // = OUT_K
constant constexpr uint BK = 32;

kernel void conv_transpose2d_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  w       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   N       [[buffer(3)]],
    constant     uint&   C_in    [[buffer(4)]],
    constant     uint&   H       [[buffer(5)]],
    constant     uint&   W       [[buffer(6)]],
    constant     uint&   C_out   [[buffer(7)]],
    constant     uint&   R       [[buffer(8)]],
    constant     uint&   stride  [[buffer(9)]],
    uint tid_in_tg [[thread_position_in_threadgroup]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint n_tg      [[threadgroups_per_grid]],
    uint sgid      [[simdgroup_index_in_threadgroup]])
{
    const uint H_out = (H - 1) * stride + R;   // 65
    const uint W_out = (W - 1) * stride + R;   // 65
    const uint HxW_out = H_out * W_out;
    const uint M_total = N * HxW_out;
    const uint num_m_tiles = (M_total + BM - 1) / BM;

    threadgroup float Atile[BM * BK];   // 4096 floats = 16KB
    threadgroup float Btile[BK * BN];   // 4096 floats = 16KB

    const uint sg_row = sgid >> 1;        // 0..15
    const uint sg_col = sgid & 1u;        // 0..1
    const uint row0 = sg_row * 8;
    const uint col0 = sg_col * 64;

    for (uint mtile = tg_id; mtile < num_m_tiles; mtile += n_tg) {
        const uint m_base = mtile * BM;

        simdgroup_matrix<float, 8, 8> Cacc[8];
        #pragma unroll
        for (uint i = 0; i < 8; ++i) Cacc[i] = simdgroup_matrix<float,8,8>(0.0f);

        for (uint rr = 0; rr < R_K; ++rr) {
            for (uint ss = 0; ss < S_K; ++ss) {
                const uint w_base = (rr * S_K + ss) * C_IN;
                #pragma unroll
                for (uint cc_base = 0; cc_base < C_IN; cc_base += BK) {
                    // ---- Stage Atile (BM=128 x BK=32). 4096 floats, 4/thread.
                    #pragma unroll
                    for (uint p = 0; p < 4; ++p) {
                        uint t = tid_in_tg + p * 1024;
                        uint a_row = t >> 5;
                        uint cc_lo = t & 31u;
                        uint cc = cc_base + cc_lo;
                        uint m_g = m_base + a_row;
                        float v = 0.0f;
                        if (m_g < M_total) {
                            uint q = m_g;
                            uint nn = q / HxW_out; q -= nn * HxW_out;
                            uint hh_out = q / W_out;
                            uint ww_out = q - hh_out * W_out;
                            int h_sig = int(hh_out) - int(rr);
                            int w_sig = int(ww_out) - int(ss);
                            if (h_sig >= 0 && w_sig >= 0 && ((h_sig | w_sig) & 1) == 0) {
                                uint hi = uint(h_sig) >> 1;
                                uint wi = uint(w_sig) >> 1;
                                if (hi < H && wi < W) {
                                    v = x[((nn * H + hi) * W + wi) * C_IN + cc];
                                }
                            }
                        }
                        Atile[a_row * BK + cc_lo] = v;
                    }

                    // ---- Stage Btile (BK=32 x BN=128). 4096 floats, 4/thread.
                    #pragma unroll
                    for (uint p = 0; p < 4; ++p) {
                        uint t = tid_in_tg + p * 1024;
                        uint k_lo = t >> 7;
                        uint n_in = t & 127u;
                        Btile[k_lo * BN + n_in] = w[n_in * KD + w_base + cc_base + k_lo];
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    #pragma unroll
                    for (uint k0 = 0; k0 < BK; k0 += 8) {
                        simdgroup_matrix<float, 8, 8> A;
                        simdgroup_load(A, Atile + row0 * BK + k0, BK, ulong2(0, 0));
                        #pragma unroll
                        for (uint t = 0; t < 8; ++t) {
                            uint c = col0 + t * 8;
                            simdgroup_matrix<float, 8, 8> B;
                            simdgroup_load(B, Btile + k0 * BN + c, BN, ulong2(0, 0));
                            simdgroup_multiply_accumulate(Cacc[t], A, B, Cacc[t]);
                        }
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }

        const uint m_row0 = m_base + row0;
        if (m_row0 + 8 <= M_total) {
            device float* y_ptr = y + m_row0 * OUT_K + col0;
            #pragma unroll
            for (uint t = 0; t < 8; ++t) {
                simdgroup_store(Cacc[t], y_ptr + t * 8, OUT_K, ulong2(0, 0));
            }
        } else if (m_row0 < M_total) {
            threadgroup float* scratch = Btile + sgid * 64;
            #pragma unroll
            for (uint t = 0; t < 8; ++t) {
                simdgroup_store(Cacc[t], scratch, 8, ulong2(0, 0));
                simdgroup_barrier(mem_flags::mem_threadgroup);
                uint lane = tid_in_tg & 31u;
                uint r = lane >> 3;
                uint cc = lane & 7u;
                uint m_g0 = m_base + row0 + r;
                if (m_g0 < M_total) {
                    y[m_g0 * OUT_K + col0 + t * 8 + cc] = scratch[r * 8 + cc];
                }
                uint m_g1 = m_base + row0 + r + 4;
                if (m_g1 < M_total) {
                    y[m_g1 * OUT_K + col0 + t * 8 + cc] = scratch[(r + 4) * 8 + cc];
                }
            }
        }
    }
}
