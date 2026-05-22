// conv3d implicit im2col GEMM with simdgroup_matrix MMA, fused activation+bias epilogue.
// NDHWC, k=3, stride=1. x (4,32,32,32,32), w (64,3,3,3,32), b(64) -> y (4,30,30,30,64).
// GEMM: M = 4*30^3 = 108000, N = 64, K = 27*32 = 864.
// Activation chain (post-conv): relu -> leaky(0.01) -> gelu -> sigmoid + bias.
//   For x>=0 leaky_relu is identity. So fused fn collapses to:
//     v = max(conv, 0); v = gelu_erf(v); v = sigmoid(v); v += b[k].

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

static inline float gelu_erf_poly(float x) {
    const float k = 0.70710678f;
    float z = x * k;
    float t = 1.0f / (1.0f + 0.3275911f * fabs(z));
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t)
              + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f)
              * t * exp(-z * z);
    float erfz = copysign(y, z);
    return 0.5f * x * (1.0f + erfz);
}

static inline float fused_act(float x, float bias) {
    float v = fmax(x, 0.0f);            // relu (leaky no-op since v>=0)
    v = gelu_erf_poly(v);               // gelu
    v = 1.0f / (1.0f + exp(-v));        // sigmoid (v bounded since x>=0 -> gelu>=0 -> sig in [0.5,1])
    return v + bias;
}

kernel void conv3d_multi_act_bias_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  w       [[buffer(1)]],
    device const float*  b       [[buffer(2)]],
    device       float*  y       [[buffer(3)]],
    constant     uint&   N_      [[buffer(4)]],
    constant     uint&   C       [[buffer(5)]],
    constant     uint&   D       [[buffer(6)]],
    constant     uint&   H       [[buffer(7)]],
    constant     uint&   W       [[buffer(8)]],
    constant     uint&   K_out   [[buffer(9)]],
    constant     uint&   R       [[buffer(10)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint sgid    [[simdgroup_index_in_threadgroup]],
    uint lid     [[thread_index_in_threadgroup]],
    uint n_tg    [[threadgroups_per_grid]])
{
    const uint D2 = D - R + 1;
    const uint H2 = H - R + 1;
    const uint W2 = W - R + 1;
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

        // Per-thread A-row coords (constant over k-loop).
        const uint a_row = lid / 16u;
        const uint a_col0 = lid % 16u;
        const uint m_g_A = c_row0 + a_row;
        uint a_n=0, a_d=0, a_h=0, a_w=0;
        bool a_valid_row = m_g_A < M_g;
        if (a_valid_row) {
            a_n = m_g_A / DHW2;
            uint q = m_g_A - a_n * DHW2;
            a_d = q / HW2;
            q -= a_d * HW2;
            a_h = q / W2;
            a_w = q - a_h * W2;
        }

        for (uint kt = 0; kt < num_k_tiles; ++kt) {
            const uint k0 = kt * BK;

            // Load A: 64*16 = 1024 elems, 1/thread.
            {
                const uint a_col = a_col0;
                const uint k_glob = k0 + a_col;
                const uint rd = k_glob / RRC;
                uint rem = k_glob - rd * RRC;
                const uint rh = rem / RC;
                rem -= rh * RC;
                const uint rw = rem / C;
                const uint c  = rem - rw * C;

                float v = 0.0f;
                if (a_valid_row && k_glob < K_g) {
                    const uint di = a_d + rd;
                    const uint hi = a_h + rh;
                    const uint wi = a_w + rw;
                    v = x[(((a_n * D + di) * H + hi) * W + wi) * C + c];
                }
                As[a_row * LDA + a_col] = v;
            }
            // Load B: 16 rows × 64 cols = 1024 elems.
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

        // Epilogue: fused activation + bias, store with float4.
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
                // Load 4 biases.
                float4 bv;
                if (nc + 3 < K_out) {
                    bv = *reinterpret_cast<device const float4*>(&b[nc]);
                } else {
                    bv = float4(0.0f);
                    for (uint kk = 0; kk < 4u; ++kk)
                        if (nc + kk < K_out) ((thread float*)&bv)[kk] = b[nc + kk];
                }
                v.x = fused_act(v.x, bv.x);
                v.y = fused_act(v.y, bv.y);
                v.z = fused_act(v.z, bv.z);
                v.w = fused_act(v.w, bv.w);
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
