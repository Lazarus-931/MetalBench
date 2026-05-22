// conv2d: implicit-im2col GEMM, NHWC, k=3, stride=1.
// Tile BM=64, BN=128, BK=16. 1024 threads = 32x32. Each thread 2x4 reg tile.
#include <metal_stdlib>
using namespace metal;

kernel void conv2d_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  w       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   N       [[buffer(3)]], constant     uint&   C       [[buffer(4)]],
    constant     uint&   H       [[buffer(5)]], constant     uint&   W       [[buffer(6)]],
    constant     uint&   K       [[buffer(7)]], constant     uint&   R       [[buffer(8)]],
    constant     uint&   S       [[buffer(9)]], constant     uint&   stride  [[buffer(10)]],
    uint  tid_in_tg [[thread_position_in_threadgroup]],
    uint  tg_id     [[threadgroup_position_in_grid]],
    uint  n_tg      [[threadgroups_per_grid]])
{
    constexpr uint BM = 128;
    constexpr uint BN = 128;
    constexpr uint BK = 16;
    constexpr uint TM = 4;   // rows per thread
    constexpr uint TN = 4;   // cols per thread

    const uint H2 = (H - R) / stride + 1;
    const uint W2 = (W - S) / stride + 1;
    const uint M  = N * H2 * W2;
    const uint Kd = C * R * S;
    const uint SC = S * C;

    const uint num_m_tiles = (M + BM - 1) / BM;

    threadgroup float Atile[BM * BK];   // 1024
    threadgroup float Btile[BK * BN];   // 2048

    // Thread layout for compute: 32 rows of M (each with TM=2 → BM=64) × 32 cols of N (each TN=4 → BN=128)
    const uint ty = tid_in_tg / 32;     // 0..31, M tile row group
    const uint tx = tid_in_tg % 32;     // 0..31, N tile col group

    for (uint mtile = tg_id; mtile < num_m_tiles; mtile += n_tg) {
        const uint m_base = mtile * BM;

        // Decode (n,h2,w2) for the 2 rows this thread owns.
        uint n_idx[TM], h2_idx[TM], w2_idx[TM];
        bool row_valid[TM];
        for (uint i = 0; i < TM; ++i) {
            uint m_g = m_base + ty + i * 32;
            row_valid[i] = (m_g < M);
            if (row_valid[i]) {
                uint q = m_g;
                n_idx[i]  = q / (H2 * W2); q %= (H2 * W2);
                h2_idx[i] = q / W2;
                w2_idx[i] = q % W2;
            } else {
                n_idx[i] = h2_idx[i] = w2_idx[i] = 0;
            }
        }

        float acc[TM][TN];
        for (uint i = 0; i < TM; ++i)
            for (uint j = 0; j < TN; ++j)
                acc[i][j] = 0.0f;

        for (uint k0 = 0; k0 < Kd; k0 += BK) {
            // A tile: BM*BK = 2048 entries, 1024 threads → 2 each.
            for (uint pass = 0; pass < 2; ++pass) {
                uint t = tid_in_tg + pass * 1024;
                uint a_row = t / BK;            // 0..127
                uint a_col = t % BK;            // 0..15
                uint k_glob = k0 + a_col;
                uint rr = k_glob / SC;
                uint rem = k_glob - rr * SC;
                uint ss = rem / C;
                uint c  = rem - ss * C;

                uint mrow_g = m_base + a_row;
                float v = 0.0f;
                if (mrow_g < M) {
                    uint q = mrow_g;
                    uint nn  = q / (H2 * W2); q %= (H2 * W2);
                    uint hh2 = q / W2;
                    uint ww2 = q % W2;
                    uint hi = hh2 * stride + rr;
                    uint wi = ww2 * stride + ss;
                    v = x[((nn * H + hi) * W + wi) * C + c];
                }
                Atile[a_row * BK + a_col] = v;
            }

            // B tile: BK*BN = 2048 → 2 each.
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

            // Compute: TM=4 rows (ty + i*32) × TN=4 cols (tx*4 + j)
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

        // Write out
        uint col0 = tx * TN;
        for (uint i = 0; i < TM; ++i) {
            if (row_valid[i]) {
                uint y_base = ((n_idx[i] * H2 + h2_idx[i]) * W2 + w2_idx[i]) * K;
                y[y_base + col0 + 0] = acc[i][0];
                y[y_base + col0 + 1] = acc[i][1];
                y[y_base + col0 + 2] = acc[i][2];
                y[y_base + col0 + 3] = acc[i][3];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
