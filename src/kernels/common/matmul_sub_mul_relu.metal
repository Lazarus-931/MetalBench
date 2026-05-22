// matmul_sub_mul_relu: y = ReLU((x @ w - sub_val) * mul_val).
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM = 64, BN = 64, BK = 16;
constant constexpr uint SM = 16, SN = 32;
constant constexpr uint SIMDS_N = BN / SN;
constant constexpr uint MMA_M = SM / 8, MMA_N = SN / 8;
constant constexpr uint PAD = 4, LDA = BK + PAD, LDB = BN + PAD;
constant constexpr uint TILES_N = 256 / BN;
constant constexpr uint ACTIVE_TG = TILES_N * (256 / BM);

kernel void matmul_sub_mul_relu_f32(
    device const float* X       [[buffer(0)]],
    device const float* W       [[buffer(1)]],
    device       float* Y       [[buffer(2)]],
    constant     uint&  M       [[buffer(3)]],
    constant     uint&  N       [[buffer(4)]],
    constant     uint&  K       [[buffer(5)]],
    constant     float& sub_val [[buffer(6)]],
    constant     float& mul_val [[buffer(7)]],
    uint tgid_lin               [[threadgroup_position_in_grid]],
    uint sgid                   [[simdgroup_index_in_threadgroup]],
    uint lid                    [[thread_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]])
{
    if (tgid_lin >= ACTIVE_TG) return;
    const uint tx = tgid_lin % TILES_N;
    const uint ty = tgid_lin / TILES_N;

    constexpr uint AS_SZ = 2 * BM * LDA;        // 2560
    constexpr uint BS_SZ = 2 * BK * LDB;        // 2176
    constexpr uint CS_SZ = BM * BN;             // 4096
    constexpr uint SCRATCH = (AS_SZ + BS_SZ > CS_SZ) ? (AS_SZ + BS_SZ) : CS_SZ;
    threadgroup float scratch[SCRATCH];
    threadgroup float (*As)[BM * LDA] = reinterpret_cast<threadgroup float(*)[BM * LDA]>(&scratch[0]);
    threadgroup float (*Bs)[BK * LDB] = reinterpret_cast<threadgroup float(*)[BK * LDB]>(&scratch[AS_SZ]);
    threadgroup float* Cs = &scratch[0];

    const uint sm = sgid / SIMDS_N;
    const uint sn = sgid % SIMDS_N;
    const uint c_row0 = ty * BM;
    const uint c_col0 = tx * BN;

    const uint a_row = lid / 4;
    const uint a_c4  = lid % 4;
    const uint b_row = lid / 16;
    const uint b_c4  = lid % 16;

    simdgroup_matrix<float, 8, 8> C_acc[MMA_M][MMA_N];
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            C_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    {
        *reinterpret_cast<threadgroup float4*>(&As[0][a_row * LDA + a_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&X[(c_row0 + a_row) * K + a_c4 * 4]);
        *reinterpret_cast<threadgroup float4*>(&Bs[0][b_row * LDB + b_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&W[b_row * N + c_col0 + b_c4 * 4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_k_tiles = K / BK;
    uint buf = 0;

    for (uint kt = 0; kt < num_k_tiles - 1; ++kt) {
        const uint next = 1 - buf;
        const uint k0_nxt = (kt + 1) * BK;

        *reinterpret_cast<threadgroup float4*>(&As[next][a_row * LDA + a_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&X[(c_row0 + a_row) * K + k0_nxt + a_c4 * 4]);
        *reinterpret_cast<threadgroup float4*>(&Bs[next][b_row * LDB + b_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&W[(k0_nxt + b_row) * N + c_col0 + b_c4 * 4]);

        #pragma unroll
        for (uint kc = 0; kc < BK; kc += 8) {
            simdgroup_matrix<float, 8, 8> A_blk[MMA_M], B_blk[MMA_N];
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                simdgroup_load(A_blk[i], &As[buf][(sm * SM + i * 8) * LDA + kc], LDA);
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_load(B_blk[j], &Bs[buf][kc * LDB + sn * SN + j * 8], LDB);
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                #pragma unroll
                for (uint j = 0; j < MMA_N; ++j)
                    simdgroup_multiply_accumulate(C_acc[i][j], A_blk[i], B_blk[j], C_acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf = next;
    }

    #pragma unroll
    for (uint kc = 0; kc < BK; kc += 8) {
        simdgroup_matrix<float, 8, 8> A_blk[MMA_M], B_blk[MMA_N];
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            simdgroup_load(A_blk[i], &As[buf][(sm * SM + i * 8) * LDA + kc], LDA);
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            simdgroup_load(B_blk[j], &Bs[buf][kc * LDB + sn * SN + j * 8], LDB);
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_multiply_accumulate(C_acc[i][j], A_blk[i], B_blk[j], C_acc[i][j]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            simdgroup_store(C_acc[i][j],
                            &Cs[(sm * SM + i * 8) * BN + (sn * SN + j * 8)],
                            BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_f4 = BM * BN / 4;
    for (uint idx = lid; idx < total_f4; idx += 256) {
        uint r = (idx * 4) / BN;
        uint c = (idx * 4) % BN;
        float4 v = *reinterpret_cast<threadgroup float4*>(&Cs[r * BN + c]);
        float4 out = (v - sub_val) * mul_val;
        out = fmax(out, float4(0.0f));
        *reinterpret_cast<device float4*>(&Y[(c_row0 + r) * N + c_col0 + c]) = out;
    }
}
