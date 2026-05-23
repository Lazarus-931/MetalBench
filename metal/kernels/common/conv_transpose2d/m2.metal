// conv_transpose2d M2-optimized: NHWC, R=S=3, stride configurable
// (fast path stride==2). 128x128 tile, BK=16, hand-coded outer-product MMA.
//
// M2 wins vs default.metal:
//   - Per-row output decomposition cached in TG memory (computed once per tile,
//     avoiding redundant int div/mod inside the k-loop).
//   - Per-k weight decomposition cached in TG memory.
//   - stride==2 power-of-2 fast path (parity test == lsb test, hi=h_sig>>1).
//   - C_in==64 power-of-2 fast path (modulo, divide via shift+mask).
//   - float4 vectorized B-tile reads + float4 epilogue stores.
//   - Streamed-A inner loop (A read from TG mem; no register A tile).
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(1024)]]
kernel void conv_transpose2d_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  w       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   N_      [[buffer(3)]],
    constant     uint&   C_in    [[buffer(4)]],
    constant     uint&   H       [[buffer(5)]],
    constant     uint&   W       [[buffer(6)]],
    constant     uint&   C_out   [[buffer(7)]],
    constant     uint&   R       [[buffer(8)]],
    constant     uint&   stride_ [[buffer(9)]],
    uint  tid_in_tg [[thread_position_in_threadgroup]],
    uint  tg_id     [[threadgroup_position_in_grid]],
    uint  n_tg      [[threadgroups_per_grid]])
{
    constexpr uint BM = 128;
    constexpr uint BN = 128;
    constexpr uint BK = 16;
    constexpr uint TM = 4;
    constexpr uint TN = 4;

    const uint H_out = (H - 1) * stride_ + R;
    const uint W_out = (W - 1) * stride_ + R;
    const uint HW_out = H_out * W_out;
    const uint M_g = N_ * HW_out;
    const uint SC = R * C_in;
    const uint Kd = R * SC;
    const uint num_m_tiles = (M_g + BM - 1) / BM;

    threadgroup float Atile[BM * BK];       // 8192 B
    threadgroup float Btile[BK * BN];       // 8192 B
    threadgroup ushort Row_nn[BM];
    threadgroup ushort Row_h[BM];
    threadgroup ushort Row_w[BM];
    threadgroup uchar  Row_valid[BM];
    threadgroup uchar  K_rr[BK];
    threadgroup uchar  K_ss[BK];
    threadgroup ushort K_c[BK];

    const uint ty = tid_in_tg / 32;
    const uint tx = tid_in_tg & 31u;

    const bool c_in_pow2_64 = (C_in == 64);
    const bool stride_2 = (stride_ == 2);

    for (uint mtile = tg_id; mtile < num_m_tiles; mtile += n_tg) {
        const uint m_base = mtile * BM;

        // Per-row metadata (BM=128, use first 128 threads).
        if (tid_in_tg < BM) {
            const uint m_g = m_base + tid_in_tg;
            if (m_g < M_g) {
                const uint nn = m_g / HW_out;
                uint q = m_g - nn * HW_out;
                const uint hh = q / W_out;
                const uint ww = q - hh * W_out;
                Row_nn[tid_in_tg] = (ushort)nn;
                Row_h[tid_in_tg]  = (ushort)hh;
                Row_w[tid_in_tg]  = (ushort)ww;
                Row_valid[tid_in_tg] = 1;
            } else {
                Row_valid[tid_in_tg] = 0;
            }
        }

        // Per-thread row metadata for epilogue.
        uint n_idx[TM], h_out_idx[TM], w_out_idx[TM];
        bool row_valid_reg[TM];
        for (uint i = 0; i < TM; ++i) {
            const uint m_g = m_base + ty + i * 32u;
            row_valid_reg[i] = (m_g < M_g);
            if (row_valid_reg[i]) {
                uint q = m_g;
                n_idx[i]     = q / HW_out; q -= n_idx[i] * HW_out;
                h_out_idx[i] = q / W_out;
                w_out_idx[i] = q - h_out_idx[i] * W_out;
            }
        }

        float acc[TM][TN];
        for (uint i = 0; i < TM; ++i)
            for (uint j = 0; j < TN; ++j) acc[i][j] = 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k0 = 0; k0 < Kd; k0 += BK) {
            // Per-k decomposition (BK=16, first 16 threads).
            if (tid_in_tg < BK) {
                const uint k_glob = k0 + tid_in_tg;
                uint rr, ss, c;
                if (c_in_pow2_64) {
                    c = k_glob & 63u;
                    const uint sc_idx = k_glob >> 6;
                    rr = sc_idx / R;
                    ss = sc_idx - rr * R;
                } else {
                    rr = k_glob / SC;
                    uint rem = k_glob - rr * SC;
                    ss = rem / C_in;
                    c  = rem - ss * C_in;
                }
                K_rr[tid_in_tg] = (uchar)rr;
                K_ss[tid_in_tg] = (uchar)ss;
                K_c[tid_in_tg]  = (ushort)c;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Stage A: BM*BK = 128*16 = 2048 elements / 1024 threads = 2 per thread.
            // Thread t → (row = t/BK + pass*64, col = t%BK).
            {
                const uint a_col = tid_in_tg & (BK - 1);   // 0..15
                const uint a_row0 = tid_in_tg / BK;        // 0..63
                #pragma unroll
                for (uint pass = 0; pass < 2u; ++pass) {
                    const uint a_row = a_row0 + pass * 64u;
                    float v = 0.0f;
                    if (Row_valid[a_row]) {
                        const uint nn = (uint)Row_nn[a_row];
                        const uint hh_out = (uint)Row_h[a_row];
                        const uint ww_out = (uint)Row_w[a_row];
                        const uint rr = (uint)K_rr[a_col];
                        const uint ss = (uint)K_ss[a_col];
                        const uint c  = (uint)K_c[a_col];
                        const int h_sig = int(hh_out) - int(rr);
                        const int w_sig = int(ww_out) - int(ss);
                        bool ok;
                        uint hi, wi;
                        if (stride_2) {
                            ok = (h_sig >= 0) && (w_sig >= 0)
                                 && ((h_sig & 1) == 0) && ((w_sig & 1) == 0);
                            hi = uint(h_sig) >> 1;
                            wi = uint(w_sig) >> 1;
                        } else {
                            ok = (h_sig >= 0) && (w_sig >= 0)
                                 && ((h_sig % int(stride_)) == 0)
                                 && ((w_sig % int(stride_)) == 0);
                            hi = uint(h_sig) / stride_;
                            wi = uint(w_sig) / stride_;
                        }
                        if (ok && hi < H && wi < W) {
                            v = x[((nn * H + hi) * W + wi) * C_in + c];
                        }
                    }
                    Atile[a_row * BK + a_col] = v;
                }
            }

            // Stage B: BK*BN = 16*128 = 2048 elements / 1024 = 2 per thread.
            {
                const uint t0 = tid_in_tg;
                {
                    const uint k_in = t0 / BN;          // 0..15
                    const uint n_in = t0 - k_in * BN;   // 0..127
                    Btile[k_in * BN + n_in] = w[n_in * Kd + (k0 + k_in)];
                }
                const uint t1 = tid_in_tg + 1024u;
                if (t1 < BK * BN) {
                    const uint k_in = t1 / BN;
                    const uint n_in = t1 - k_in * BN;
                    Btile[k_in * BN + n_in] = w[n_in * Kd + (k0 + k_in)];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute: TM*TN outer product, BK=16 steps.
            const uint col_base = tx * TN;
            #pragma unroll
            for (uint kk = 0; kk < BK; ++kk) {
                float4 bv = *((threadgroup const float4*)(&Btile[kk * BN + col_base]));
                #pragma unroll
                for (uint i = 0; i < TM; ++i) {
                    float a = Atile[(ty + i * 32u) * BK + kk];
                    acc[i][0] = fma(a, bv.x, acc[i][0]);
                    acc[i][1] = fma(a, bv.y, acc[i][1]);
                    acc[i][2] = fma(a, bv.z, acc[i][2]);
                    acc[i][3] = fma(a, bv.w, acc[i][3]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Epilogue: float4 stores.
        const uint col0 = tx * TN;
        for (uint i = 0; i < TM; ++i) {
            if (row_valid_reg[i]) {
                const uint y_base = ((n_idx[i] * H_out + h_out_idx[i]) * W_out + w_out_idx[i]) * C_out;
                float4 out = float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
                *((device float4*)(&y[y_base + col0])) = out;
            }
        }
    }
}
