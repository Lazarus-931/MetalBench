// conv_transpose2d_sub_tanh: fused 3x3 transposed conv (NHWC, stride=2)
// + sub_val + stable tanh, via implicit-im2col GEMM with simdgroup_matrix MMA.
// Each TG processes 2 consecutive m-tiles, sharing per-kt B (weight) load.
// Grid-agnostic: TG strides over m-tile groups via tg_id.
//
// Shapes: x (N=8, H=32, W=32, C=64) ; w (K_out=128, R=3, S=3, C=64) ;
//         y (N, H_out=65, W_out=65, K_out=128) ; stride=2.
//
// GEMM mapping: M = N*H_out*W_out = 33800 ; N_dim = K_out = 128 ; K = R*S*C = 576.
// x[m, k] := x[n, h_in, w_in, c] iff (h_out-r) and (w_out-s) are both even and
//            input coord is in-range; else 0.  Parity folded into A-load.
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

inline float safe_tanh(float v) {
    return precise::tanh(clamp(v, -30.0f, 30.0f));
}

kernel void conv_transpose2d_sub_tanh_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  w       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   N_      [[buffer(3)]],
    constant     uint&   C_in    [[buffer(4)]],
    constant     uint&   H       [[buffer(5)]],
    constant     uint&   W       [[buffer(6)]],
    constant     uint&   K_out   [[buffer(7)]],
    constant     uint&   R       [[buffer(8)]],
    constant     uint&   stride  [[buffer(9)]],
    constant     float&  sub_val [[buffer(10)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint sgid    [[simdgroup_index_in_threadgroup]],
    uint lid     [[thread_index_in_threadgroup]],
    uint n_tg    [[threadgroups_per_grid]])
{
    const uint H_out = (H - 1) * stride + R;  // 65
    const uint W_out = (W - 1) * stride + R;  // 65
    const uint HW_out = H_out * W_out;
    const uint M_g = N_ * HW_out;
    const uint S   = R;
    const uint SC  = S * C_in;
    const uint K_g = R * SC;
    const uint num_m_tiles = (M_g + BM - 1) / BM;
    const uint num_groups = (num_m_tiles + 1u) / 2u;

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

            // Load A0, A1 (BM x BK each): 1 element/thread.
            {
                const uint a_row = lid / BK;
                const uint a_col = lid & (BK - 1);
                const uint k_glob = k0 + a_col;
                float v0 = 0.0f, v1 = 0.0f;
                if (k_glob < K_g) {
                    const uint rr = k_glob / SC;
                    const uint rem = k_glob - rr * SC;
                    const uint ss = rem / C_in;
                    const uint c  = rem - ss * C_in;

                    // Tile 0
                    {
                        const uint mg = c_row0_0 + a_row;
                        if (mg < M_g) {
                            const uint n_idx = mg / HW_out;
                            const uint hw    = mg - n_idx * HW_out;
                            const uint h_out = hw / W_out;
                            const uint w_out = hw - h_out * W_out;
                            const int hd = int(h_out) - int(rr);
                            const int wd = int(w_out) - int(ss);
                            if (hd >= 0 && wd >= 0 && ((hd | wd) & 1) == 0) {
                                const uint h_in = uint(hd) >> 1;
                                const uint w_in = uint(wd) >> 1;
                                if (h_in < H && w_in < W) {
                                    v0 = x[((n_idx * H + h_in) * W + w_in) * C_in + c];
                                }
                            }
                        }
                    }
                    // Tile 1
                    if (has1) {
                        const uint mg = c_row0_1 + a_row;
                        if (mg < M_g) {
                            const uint n_idx = mg / HW_out;
                            const uint hw    = mg - n_idx * HW_out;
                            const uint h_out = hw / W_out;
                            const uint w_out = hw - h_out * W_out;
                            const int hd = int(h_out) - int(rr);
                            const int wd = int(w_out) - int(ss);
                            if (hd >= 0 && wd >= 0 && ((hd | wd) & 1) == 0) {
                                const uint h_in = uint(hd) >> 1;
                                const uint w_in = uint(wd) >> 1;
                                if (h_in < H && w_in < W) {
                                    v1 = x[((n_idx * H + h_in) * W + w_in) * C_in + c];
                                }
                            }
                        }
                    }
                }
                As0[a_row * LDA + a_col] = v0;
                As1[a_row * LDA + a_col] = v1;
            }

            // Load B (BK x BN): 2 elements/thread.
            // B[k, kk] = w[kk, k_glob] where w is stored (K_out, R, S, C_in) contiguous.
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

        // Epilogue tile 0: (sum - sub_val) -> safe_tanh
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_store(C0[i][j], &Cs[(sm * SM + i * 8u) * BN + sn * SN + j * 8u], BN);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint rs = 0; rs < 2; ++rs) {
            const uint r_local = (lid / 32u) + rs * 32u;
            const uint cq = lid & 31u;
            const uint c = cq * 4u;
            const uint m_g = c_row0_0 + r_local;
            if (m_g < M_g) {
                float4 v  = *reinterpret_cast<threadgroup float4*>(&Cs[r_local * BN + c]);
                v -= float4(sub_val);
                v = clamp(v, float4(-15.0f), float4(15.0f));
                v = tanh(v);
                const uint n_idx = m_g / HW_out;
                const uint hw    = m_g - n_idx * HW_out;
                const uint h_out = hw / W_out;
                const uint w_out = hw - h_out * W_out;
                *reinterpret_cast<device float4*>(&y[((n_idx * H_out + h_out) * W_out + w_out) * K_out + c]) = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (has1) {
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                #pragma unroll
                for (uint j = 0; j < MMA_N; ++j)
                    simdgroup_store(C1[i][j], &Cs[(sm * SM + i * 8u) * BN + sn * SN + j * 8u], BN);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (uint rs = 0; rs < 2; ++rs) {
                const uint r_local = (lid / 32u) + rs * 32u;
                const uint cq = lid & 31u;
                const uint c = cq * 4u;
                const uint m_g = c_row0_1 + r_local;
                if (m_g < M_g) {
                    float4 v  = *reinterpret_cast<threadgroup float4*>(&Cs[r_local * BN + c]);
                    v -= float4(sub_val);
                    v = clamp(v, float4(-15.0f), float4(15.0f));
                    v = tanh(v);
                    const uint n_idx = m_g / HW_out;
                    const uint hw    = m_g - n_idx * HW_out;
                    const uint h_out = hw / W_out;
                    const uint w_out = hw - h_out * W_out;
                    *reinterpret_cast<device float4*>(&y[((n_idx * H_out + h_out) * W_out + w_out) * K_out + c]) = v;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
