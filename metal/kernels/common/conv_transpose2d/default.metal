// conv_transpose2d: implicit-im2col GEMM. NHWC, R=S=3, stride=2.
#include <metal_stdlib>
using namespace metal;

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
    uint  tid_in_tg [[thread_position_in_threadgroup]],
    uint  tg_id     [[threadgroup_position_in_grid]],
    uint  n_tg      [[threadgroups_per_grid]])
{
    constexpr uint BM = 128;
    constexpr uint BN = 128;
    constexpr uint BK = 16;
    constexpr uint TM = 4;
    constexpr uint TN = 4;

    const uint H_out = (H - 1) * stride + R;
    const uint W_out = (W - 1) * stride + R;
    const uint M     = N * H_out * W_out;
    const uint S     = R;
    const uint Kd    = R * S * C_in;
    const uint SC    = S * C_in;

    const uint num_m_tiles = (M + BM - 1) / BM;

    threadgroup float Atile[BM * BK];   // 2048
    threadgroup float Btile[BK * BN];   // 2048

    const uint ty = tid_in_tg / 32;
    const uint tx = tid_in_tg % 32;

    for (uint mtile = tg_id; mtile < num_m_tiles; mtile += n_tg) {
        const uint m_base = mtile * BM;

        uint n_idx[TM], h_out_idx[TM], w_out_idx[TM];
        bool row_valid[TM];
        for (uint i = 0; i < TM; ++i) {
            uint m_g = m_base + ty + i * 32;
            row_valid[i] = (m_g < M);
            if (row_valid[i]) {
                uint q = m_g;
                n_idx[i]     = q / (H_out * W_out); q %= (H_out * W_out);
                h_out_idx[i] = q / W_out;
                w_out_idx[i] = q % W_out;
            } else {
                n_idx[i] = h_out_idx[i] = w_out_idx[i] = 0;
            }
        }

        float acc[TM][TN];
        for (uint i = 0; i < TM; ++i)
            for (uint j = 0; j < TN; ++j)
                acc[i][j] = 0.0f;

        for (uint k0 = 0; k0 < Kd; k0 += BK) {
            for (uint pass = 0; pass < 2; ++pass) {
                uint t = tid_in_tg + pass * 1024;
                uint a_row = t / BK;
                uint a_col = t % BK;
                uint k_glob = k0 + a_col;
                uint rr = k_glob / SC;
                uint rem = k_glob - rr * SC;
                uint ss = rem / C_in;
                uint c  = rem - ss * C_in;

                uint mrow_g = m_base + a_row;
                float v = 0.0f;
                if (mrow_g < M) {
                    uint q = mrow_g;
                    uint nn  = q / (H_out * W_out); q %= (H_out * W_out);
                    uint hh_out = q / W_out;
                    uint ww_out = q % W_out;
                    int h_sig = int(hh_out) - int(rr);
                    int w_sig = int(ww_out) - int(ss);
                    if (h_sig >= 0 && w_sig >= 0
                        && (h_sig % int(stride)) == 0
                        && (w_sig % int(stride)) == 0) {
                        uint hi = uint(h_sig) / stride;
                        uint wi = uint(w_sig) / stride;
                        if (hi < H && wi < W) {
                            v = x[((nn * H + hi) * W + wi) * C_in + c];
                        }
                    }
                }
                Atile[a_row * BK + a_col] = v;
            }

            {
                uint t0 = tid_in_tg;
                uint t1 = tid_in_tg + 1024;
                {
                    uint k_in = t0 / BN;
                    uint n_in = t0 % BN;
                    Btile[k_in * BN + n_in] = w[n_in * Kd + (k0 + k_in)];
                }
                if (t1 < BK * BN) {
                    uint k_in = t1 / BN;
                    uint n_in = t1 % BN;
                    Btile[k_in * BN + n_in] = w[n_in * Kd + (k0 + k_in)];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint col_base = tx * TN;
            for (uint kk = 0; kk < BK; ++kk) {
                float b0 = Btile[kk * BN + col_base + 0];
                float b1 = Btile[kk * BN + col_base + 1];
                float b2 = Btile[kk * BN + col_base + 2];
                float b3 = Btile[kk * BN + col_base + 3];
                #pragma unroll
                for (uint i = 0; i < TM; ++i) {
                    float a = Atile[(ty + i * 32) * BK + kk];
                    acc[i][0] = fma(a, b0, acc[i][0]);
                    acc[i][1] = fma(a, b1, acc[i][1]);
                    acc[i][2] = fma(a, b2, acc[i][2]);
                    acc[i][3] = fma(a, b3, acc[i][3]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint col0 = tx * TN;
        for (uint i = 0; i < TM; ++i) {
            if (row_valid[i]) {
                uint y_base = ((n_idx[i] * H_out + h_out_idx[i]) * W_out + w_out_idx[i]) * C_out;
                y[y_base + col0 + 0] = acc[i][0];
                y[y_base + col0 + 1] = acc[i][1];
                y[y_base + col0 + 2] = acc[i][2];
                y[y_base + col0 + 3] = acc[i][3];
            }
        }
    }
}
