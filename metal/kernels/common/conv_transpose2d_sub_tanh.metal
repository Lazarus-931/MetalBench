// conv_transpose2d_sub_tanh: fused 3x3 transposed conv (NHWC, stride=2)
// + sub_val + stable tanh, via implicit-im2col GEMM with simdgroup_matrix MMA.
//
// Shapes: x (N=8, H=32, W=32, C=64) ; w (K_out=128, R=3, S=3, C=64) ;
//         y (N, H_out=65, W_out=65, K_out=128) ; stride=2.
//
// GEMM mapping: M = N*H_out*W_out ; N_dim = K_out ; K = R*S*C.
// x[m, k] := x[n, h_in, w_in, c] iff (h_out-r) and (w_out-s) are both even and
//            input coord is in-range; else 0.
//
// Tile: BM=64, BN=64, BK=16. TG=512 threads = 16 simdgroups (SIMDS_M=4, SIMDS_N=4).
// One m-tile per TG iteration; TG strides over all m-tiles.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM = 64, BN = 64, BK = 16;
constant constexpr uint SM = 16, SN = 16;
constant constexpr uint SIMDS_M = BM / SM;   // 4
constant constexpr uint SIMDS_N = BN / SN;   // 4
constant constexpr uint MMA_M = SM / 8;      // 2
constant constexpr uint MMA_N = SN / 8;      // 2
constant constexpr uint TG_THREADS = 512;
constant constexpr uint PAD = 4;
constant constexpr uint LDA = BK + PAD;      // 20
constant constexpr uint LDB = BN + PAD;      // 68

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
    const uint H_out = (H - 1) * stride + R;
    const uint W_out = (W - 1) * stride + R;
    const uint HW_out = H_out * W_out;
    const uint M_g = N_ * HW_out;
    const uint S   = R;
    const uint SC  = S * C_in;
    const uint K_g = R * SC;
    const uint num_m_tiles = (M_g + BM - 1) / BM;
    const uint num_n_tiles = (K_out + BN - 1) / BN;
    const uint num_tiles = num_m_tiles * num_n_tiles;

    // scratch usage:
    //  As: BM * LDA = 64 * 20 = 1280 floats
    //  Bs: BK * LDB = 16 * 68 = 1088 floats
    //  Cs (reused after MMA): BM * BN = 64 * 64 = 4096 floats
    // Max = 4096 floats = 16 KB. Use a union via shared base.
    threadgroup float scratch[BM * BN];
    threadgroup float* As = scratch;
    threadgroup float* Bs = scratch + BM * LDA;
    threadgroup float* Cs = scratch;

    const uint sm = sgid / SIMDS_N;
    const uint sn = sgid % SIMDS_N;

    for (uint tile = tg_id; tile < num_tiles; tile += n_tg) {
        const uint mt = tile / num_n_tiles;
        const uint nt = tile - mt * num_n_tiles;
        const uint c_row0 = mt * BM;
        const uint c_col0 = nt * BN;

        simdgroup_matrix<float, 8, 8> C[MMA_M][MMA_N];
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                C[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

        const uint num_k_tiles = (K_g + BK - 1) / BK;

        for (uint kt = 0; kt < num_k_tiles; ++kt) {
            const uint k0 = kt * BK;

            // Load A (BM x BK = 1024 elements, 2 per thread).
            #pragma unroll
            for (uint p = 0; p < 2; ++p) {
                const uint t = lid + p * TG_THREADS;
                const uint a_row = t / BK;
                const uint a_col = t & (BK - 1);
                const uint k_glob = k0 + a_col;
                float v0 = 0.0f;
                if (k_glob < K_g && a_row < BM) {
                    const uint rr = k_glob / SC;
                    const uint rem = k_glob - rr * SC;
                    const uint ss = rem / C_in;
                    const uint c  = rem - ss * C_in;
                    const uint mg = c_row0 + a_row;
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
                As[a_row * LDA + a_col] = v0;
            }

            // Load B (BK x BN = 1024 elements, 2 per thread).
            // w stored (K_out, R, S, C_in) contiguous; B[k_g, kk] = w[c_col0+kk, k_g].
            #pragma unroll
            for (uint p = 0; p < 2; ++p) {
                const uint t = lid + p * TG_THREADS;
                const uint b_col = t / BK;
                const uint b_row = t & (BK - 1);
                const uint kg = k0 + b_row;
                const uint kk = c_col0 + b_col;
                float v = 0.0f;
                if (kg < K_g && kk < K_out) v = w[kk * K_g + kg];
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
                        simdgroup_multiply_accumulate(C[i][j], A_blk[i], B_blk[j], C[i][j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Store C to threadgroup, then apply sub + tanh and write to y.
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_store(C[i][j], &Cs[(sm * SM + i * 8u) * BN + sn * SN + j * 8u], BN);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // BM*BN = 4096 elements / 512 threads = 8 elements per thread = 2 float4 per thread.
        #pragma unroll
        for (uint p = 0; p < 2; ++p) {
            const uint t = lid + p * TG_THREADS;        // 0..1023
            const uint r_local = t / (BN / 4u);          // BN/4=16 ; r_local 0..63
            const uint cq = t & ((BN / 4u) - 1u);
            const uint c = cq * 4u;
            const uint m_g = c_row0 + r_local;
            const uint k_g = c_col0 + c;
            if (m_g < M_g && k_g < K_out) {
                float4 v = *reinterpret_cast<threadgroup float4*>(&Cs[r_local * BN + c]);
                v -= float4(sub_val);
                v = clamp(v, float4(-15.0f), float4(15.0f));
                v = tanh(v);
                const uint n_idx = m_g / HW_out;
                const uint hw    = m_g - n_idx * HW_out;
                const uint h_out = hw / W_out;
                const uint w_out = hw - h_out * W_out;
                *reinterpret_cast<device float4*>(&y[((n_idx * H_out + h_out) * W_out + w_out) * K_out + k_g]) = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
