// matmul_sub_mul_relu: y = ReLU((x @ w - sub_val) * mul_val).
// Optimized for M=N=K=256 on M4: 8x8 simdgroup MMA tiles, BM=BN=64, BK=32,
// double-buffered float4 loads, fused (sub + mul + ReLU) epilogue.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM = 64, BN = 64, BK = 32;
constant constexpr uint SM = 16, SN = 32;
constant constexpr uint SIMDS_N = BN / SN;          // 2
constant constexpr uint MMA_M = SM / 8, MMA_N = SN / 8;  // 2, 4
constant constexpr uint LDA = BK, LDB = BN;
constant constexpr uint TILES_N = 256 / BN;          // 4

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
    const uint tx = tgid_lin % TILES_N;
    const uint ty = tgid_lin / TILES_N;

    // Double-buffered A, B; Cs aliases. 32KB tg_mem.
    constexpr uint AS_SZ = 2 * BM * LDA;   // 4096
    constexpr uint BS_SZ = 2 * BK * LDB;   // 4096
    constexpr uint SCRATCH = AS_SZ + BS_SZ; // 8192 floats = 32KB
    threadgroup float scratch[SCRATCH];
    threadgroup float (*As)[BM * LDA] = reinterpret_cast<threadgroup float(*)[BM * LDA]>(&scratch[0]);
    threadgroup float (*Bs)[BK * LDB] = reinterpret_cast<threadgroup float(*)[BK * LDB]>(&scratch[AS_SZ]);
    threadgroup float* Cs = &scratch[0];

    const uint sm = sgid / SIMDS_N;
    const uint sn = sgid % SIMDS_N;
    const uint c_row0 = ty * BM;
    const uint c_col0 = tx * BN;

    const uint a_row = lid / 8;
    const uint a_c4  = lid % 8;
    const uint b_row = lid / 16;
    const uint b_c4  = lid % 16;

    simdgroup_matrix<float, 8, 8> C_acc[MMA_M][MMA_N];
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            C_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    // Prologue: tile 0.
    #pragma unroll
    for (uint p = 0; p < 2; ++p) {
        uint r = a_row + p * 32;
        *reinterpret_cast<threadgroup float4*>(&As[0][r * LDA + a_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&X[(c_row0 + r) * K + a_c4 * 4]);
    }
    #pragma unroll
    for (uint p = 0; p < 2; ++p) {
        uint r = b_row + p * 16;
        *reinterpret_cast<threadgroup float4*>(&Bs[0][r * LDB + b_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&W[r * N + c_col0 + b_c4 * 4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_k_tiles = K / BK;   // 8
    uint buf = 0;

    for (uint kt = 0; kt < num_k_tiles - 1; ++kt) {
        const uint next = 1 - buf;
        const uint k0_nxt = (kt + 1) * BK;

        #pragma unroll
        for (uint p = 0; p < 2; ++p) {
            uint r = a_row + p * 32;
            *reinterpret_cast<threadgroup float4*>(&As[next][r * LDA + a_c4 * 4]) =
                *reinterpret_cast<const device float4*>(&X[(c_row0 + r) * K + k0_nxt + a_c4 * 4]);
        }
        #pragma unroll
        for (uint p = 0; p < 2; ++p) {
            uint r = b_row + p * 16;
            *reinterpret_cast<threadgroup float4*>(&Bs[next][r * LDB + b_c4 * 4]) =
                *reinterpret_cast<const device float4*>(&W[(k0_nxt + r) * N + c_col0 + b_c4 * 4]);
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
        buf = next;
    }

    // Last K-tile compute.
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

    // Stage C tiles in threadgroup (aliased over As/Bs).
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            simdgroup_store(C_acc[i][j],
                            &Cs[(sm * SM + i * 8) * BN + (sn * SN + j * 8)],
                            BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Fused epilogue: sub, mul, ReLU; vector store.
    const uint total_f4 = BM * BN / 4;       // 1024
    #pragma unroll
    for (uint idx = lid; idx < total_f4; idx += 256) {
        uint r = (idx * 4) / BN;
        uint c = (idx * 4) % BN;
        float4 v = *reinterpret_cast<threadgroup float4*>(&Cs[r * BN + c]);
        float4 out = fmax((v - sub_val) * mul_val, float4(0.0f));
        *reinterpret_cast<device float4*>(&Y[(c_row0 + r) * N + c_col0 + c]) = out;
    }
}
