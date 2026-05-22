// silu_linear: y = silu(x @ W). M4 variant. M=N=K=1024, 64x64 output tile.
// Strategy: BK=32 single-buffered, PAD=0, software-pipelined device prefetch.
//   - PAD=0 keeps shared at 16 KB (BM*BK + BK*BN = 4096 f) — high TG occupancy.
//     Empirically, PAD>0 hurt 20% on M4 (extra shared + lower residency outweighs
//     any bank-conflict benefit on Apple's TG memory).
//   - Each iter prefetches the NEXT K-tile (4 float4s/thread) into registers
//     while the current MMA runs; commit-to-shared after the consumer barrier.
//     One A/B in flight at all times, no extra shared budget.
//   - 2x4 simdgroup grid (SM=32, SN=16) → 4*2 = 8 MMA tiles per simdgroup.
//   - fast::exp in the fused SiLU epilogue.
// Grid + threadgroup + output_shape unchanged.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM  = 64;
constant constexpr uint BN  = 64;
constant constexpr uint BK  = 32;
constant constexpr uint SM  = 32;
constant constexpr uint SN  = 16;
constant constexpr uint SIMDS_N = BN / SN;     // 4
constant constexpr uint MMA_M   = SM / 8;      // 4
constant constexpr uint MMA_N   = SN / 8;      // 2
constant constexpr uint PAD = 0;
constant constexpr uint LDA = BK + PAD;        // 32
constant constexpr uint LDB = BN + PAD;        // 64

kernel void silu_linear_f32(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device       float* C   [[buffer(2)]],
    constant     uint&  M   [[buffer(3)]],
    constant     uint&  N   [[buffer(4)]],
    constant     uint&  K   [[buffer(5)]],
    uint2 tgid              [[threadgroup_position_in_grid]],
    uint  sgid              [[simdgroup_index_in_threadgroup]],
    uint  lid               [[thread_index_in_threadgroup]])
{
    threadgroup float shared[BM * LDA + BK * LDB];
    threadgroup float* As = &shared[0];
    threadgroup float* Bs = &shared[BM * LDA];
    threadgroup float* Cs = &shared[0];

    const uint sm = sgid / SIMDS_N;
    const uint sn = sgid % SIMDS_N;
    const uint c_row0 = tgid.y * BM;
    const uint c_col0 = tgid.x * BN;

    const uint a_row = lid / 8u;
    const uint a_c4  = lid % 8u;
    const uint b_row = lid / 16u;
    const uint b_c4  = lid % 16u;

    // Hoisted shared offsets and per-thread row bases.
    const uint a_sh0 = a_row * LDA + a_c4 * 4u;
    const uint a_sh1 = (a_row + 32u) * LDA + a_c4 * 4u;
    const uint b_sh0 = b_row * LDB + b_c4 * 4u;
    const uint b_sh1 = (b_row + 16u) * LDB + b_c4 * 4u;
    const uint a_g0 = (c_row0 + a_row) * K + a_c4 * 4u;
    const uint a_g1 = (c_row0 + a_row + 32u) * K + a_c4 * 4u;
    const uint b_g_col = c_col0 + b_c4 * 4u;

    simdgroup_matrix<float, 8, 8> C_acc[MMA_M][MMA_N];
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            C_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    const uint num_k_tiles = K / BK;

    // Prologue: tile 0 → registers → shared.
    float4 a_lo = *reinterpret_cast<const device float4*>(&A[a_g0]);
    float4 a_hi = *reinterpret_cast<const device float4*>(&A[a_g1]);
    float4 b_lo = *reinterpret_cast<const device float4*>(&B[b_row * N + b_g_col]);
    float4 b_hi = *reinterpret_cast<const device float4*>(&B[(b_row + 16u) * N + b_g_col]);
    *reinterpret_cast<threadgroup float4*>(&As[a_sh0]) = a_lo;
    *reinterpret_cast<threadgroup float4*>(&As[a_sh1]) = a_hi;
    *reinterpret_cast<threadgroup float4*>(&Bs[b_sh0]) = b_lo;
    *reinterpret_cast<threadgroup float4*>(&Bs[b_sh1]) = b_hi;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main pipelined loop: prefetch tile (kt+1) → regs; compute tile (kt); barrier; commit.
    for (uint kt = 0; kt < num_k_tiles - 1u; ++kt) {
        const uint k0n = (kt + 1u) * BK;
        a_lo = *reinterpret_cast<const device float4*>(&A[a_g0 + k0n]);
        a_hi = *reinterpret_cast<const device float4*>(&A[a_g1 + k0n]);
        b_lo = *reinterpret_cast<const device float4*>(&B[(k0n + b_row) * N + b_g_col]);
        b_hi = *reinterpret_cast<const device float4*>(&B[(k0n + b_row + 16u) * N + b_g_col]);

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
        *reinterpret_cast<threadgroup float4*>(&As[a_sh0]) = a_lo;
        *reinterpret_cast<threadgroup float4*>(&As[a_sh1]) = a_hi;
        *reinterpret_cast<threadgroup float4*>(&Bs[b_sh0]) = b_lo;
        *reinterpret_cast<threadgroup float4*>(&Bs[b_sh1]) = b_hi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Tail: compute final tile.
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
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            simdgroup_store(C_acc[i][j],
                            &Cs[(sm * SM + i * 8u) * BN + sn * SN + j * 8u], BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Epilogue: SiLU + float4 device store. 256 threads × 4 f4 each = 1024 = 64×64.
    const uint o_row_base = lid / 16u;     // 0..15
    const uint o_c4       = lid % 16u;     // 0..15  (col base = o_c4 * 4)
    #pragma unroll
    for (uint s = 0; s < 4u; ++s) {
        uint row = o_row_base + s * 16u;
        float4 v = *reinterpret_cast<threadgroup float4*>(&Cs[row * BN + o_c4 * 4u]);
        float4 out;
        out.x = v.x / (1.0f + fast::exp(-v.x));
        out.y = v.y / (1.0f + fast::exp(-v.y));
        out.z = v.z / (1.0f + fast::exp(-v.z));
        out.w = v.w / (1.0f + fast::exp(-v.w));
        *reinterpret_cast<device float4*>(&C[(c_row0 + row) * N + c_col0 + o_c4 * 4u]) = out;
    }
}
