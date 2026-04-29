// sqr_matmul: C = A @ B  (N×N f32). simdgroup_matrix MMA, 64×64 tile,
// 8 simdgroups (4×2), double-buffered, float4 loads, BK=16.
// Constraints: N % 64 == 0, threadgroup=(256,1,1), grid=(N/64*256, N/64, 1).
// BEST_FOR: all M-series.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM           = 64;
constant constexpr uint BN           = 64;
constant constexpr uint BK           = 16;
constant constexpr uint SM           = 16;                  // per-simdgroup rows
constant constexpr uint SN           = 32;                  // per-simdgroup cols
constant constexpr uint SIMDS_M      = BM / SM;             // 4
constant constexpr uint SIMDS_N      = BN / SN;             // 2
constant constexpr uint NUM_SIMDS    = SIMDS_M * SIMDS_N;   // 8
constant constexpr uint TG_THR       = NUM_SIMDS * 32;      // 256
constant constexpr uint MMA_M        = SM / 8;              // 2
constant constexpr uint MMA_N        = SN / 8;              // 4
constant constexpr uint A_F4_PER_ROW = BK / 4;              // 4
constant constexpr uint B_F4_PER_ROW = BN / 4;              // 16

kernel void sqr_matmul_f32(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device       float* C   [[buffer(2)]],
    constant     uint&  N   [[buffer(3)]],
    uint2 tgid              [[threadgroup_position_in_grid]],
    uint  sgid              [[simdgroup_index_in_threadgroup]],
    uint  lid               [[thread_index_in_threadgroup]])
{
    threadgroup float As[2][BM * BK];
    threadgroup float Bs[2][BK * BN];

    const uint sm = sgid / SIMDS_N;
    const uint sn = sgid % SIMDS_N;
    const uint c_row0 = tgid.y * BM;
    const uint c_col0 = tgid.x * BN;

    // float4 load decomposition (1 float4 per thread per buffer).
    const uint a_row = lid / A_F4_PER_ROW;
    const uint a_c4  = lid % A_F4_PER_ROW;
    const uint b_row = lid / B_F4_PER_ROW;
    const uint b_c4  = lid % B_F4_PER_ROW;

    simdgroup_matrix<float, 8, 8> C_acc[MMA_M][MMA_N];
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            C_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    // Prologue: prime buf 0.
    {
        const device float4* a4 =
            reinterpret_cast<const device float4*>(&A[(c_row0 + a_row) * N + a_c4 * 4]);
        const device float4* b4 =
            reinterpret_cast<const device float4*>(&B[b_row * N + (c_col0 + b_c4 * 4)]);
        *reinterpret_cast<threadgroup float4*>(&As[0][a_row * BK + a_c4 * 4]) = *a4;
        *reinterpret_cast<threadgroup float4*>(&Bs[0][b_row * BN + b_c4 * 4]) = *b4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_k_tiles = N / BK;
    uint buf = 0;

    // Main loop: load (kt+1) into 1-buf while computing on buf.
    for (uint kt = 0; kt < num_k_tiles - 1; ++kt) {
        const uint next   = 1 - buf;
        const uint k0_nxt = (kt + 1) * BK;

        const device float4* a4 =
            reinterpret_cast<const device float4*>(&A[(c_row0 + a_row) * N + k0_nxt + a_c4 * 4]);
        const device float4* b4 =
            reinterpret_cast<const device float4*>(&B[(k0_nxt + b_row) * N + (c_col0 + b_c4 * 4)]);
        *reinterpret_cast<threadgroup float4*>(&As[next][a_row * BK + a_c4 * 4]) = *a4;
        *reinterpret_cast<threadgroup float4*>(&Bs[next][b_row * BN + b_c4 * 4]) = *b4;

        // 2 K-chunks of 8 (BK=16). 8 MMAs per chunk → 16 per simdgroup per K-iter.
        #pragma unroll
        for (uint kc = 0; kc < BK; kc += 8) {
            simdgroup_matrix<float, 8, 8> A_blk[MMA_M], B_blk[MMA_N];
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                simdgroup_load(A_blk[i], &As[buf][(sm * SM + i * 8) * BK + kc], BK);
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_load(B_blk[j], &Bs[buf][kc * BN + sn * SN + j * 8], BN);
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                #pragma unroll
                for (uint j = 0; j < MMA_N; ++j)
                    simdgroup_multiply_accumulate(C_acc[i][j], A_blk[i], B_blk[j], C_acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf = next;
    }

    // Epilogue: compute the final tile (loaded in last main iter).
    #pragma unroll
    for (uint kc = 0; kc < BK; kc += 8) {
        simdgroup_matrix<float, 8, 8> A_blk[MMA_M], B_blk[MMA_N];
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            simdgroup_load(A_blk[i], &As[buf][(sm * SM + i * 8) * BK + kc], BK);
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            simdgroup_load(B_blk[j], &Bs[buf][kc * BN + sn * SN + j * 8], BN);
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_multiply_accumulate(C_acc[i][j], A_blk[i], B_blk[j], C_acc[i][j]);
    }

    // Store accumulators.
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            simdgroup_store(C_acc[i][j],
                            &C[(c_row0 + sm * SM + i * 8) * N + (c_col0 + sn * SN + j * 8)],
                            N);
}
