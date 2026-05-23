// rect_matmul: C = A @ B (MxK @ KxN -> MxN f32). M=1024 N=2048 K=4096.
// M4 variant: 2x4 simdgroup grid (SM=32, SN=16), BK=16 double-buffered.
// Prefetch next K-tile while MMA-ing current tile -> overlap DRAM with FMA.
// 4 A-tiles x 2 B-tiles per kc step -> 6 simdgroup_loads / 8 MMAs.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM  = 64;
constant constexpr uint BN  = 64;
constant constexpr uint BK  = 16;
constant constexpr uint SM  = 32;
constant constexpr uint SN  = 16;
constant constexpr uint SIMDS_M = BM / SM;     // 2
constant constexpr uint SIMDS_N = BN / SN;     // 4
constant constexpr uint MMA_M   = SM / 8;      // 4
constant constexpr uint MMA_N   = SN / 8;      // 2
constant constexpr uint PAD = 4;
constant constexpr uint LDA = BK + PAD;        // 20
constant constexpr uint LDB = BN + PAD;        // 68

kernel void rect_matmul_f32(
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
    threadgroup float As[2][BM * LDA];   // 2 * 64*20 = 2560 floats = 10240 B
    threadgroup float Bs[2][BK * LDB];   // 2 * 16*68 = 2176 floats =  8704 B   total ~19 KB

    // 2x4 simdgroup layout: sm in {0,1}, sn in {0..3}.
    const uint sm = sgid / SIMDS_N;
    const uint sn = sgid % SIMDS_N;
    const uint c_row0 = tgid.y * BM;
    const uint c_col0 = tgid.x * BN;

    // A tile (64 x 16) = 1024 floats = 256 float4 / 256 threads = 1 float4/thread.
    const uint a_row = lid / 4;          // 0..63
    const uint a_c4  = lid % 4;          // 0..3
    // B tile (16 x 64) = 1024 floats = 256 float4 / 256 threads = 1 float4/thread.
    const uint b_row = lid / 16;         // 0..15
    const uint b_c4  = lid % 16;         // 0..15

    simdgroup_matrix<float, 8, 8> C_acc[MMA_M][MMA_N];
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            C_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    // Hoisted base pointers / offsets for the load streams.
    const device float* A_row_base = A + (c_row0 + a_row) * K + a_c4 * 4u;     // walks +BK each kt
    const device float* B_row_base = B + b_row * N + c_col0 + b_c4 * 4u;       // walks +BK*N each kt
    const uint as_off = a_row * LDA + a_c4 * 4u;
    const uint bs_off = b_row * LDB + b_c4 * 4u;

    // Prefetch tile 0 into buf 0.
    {
        *reinterpret_cast<threadgroup float4*>(&As[0][as_off]) =
            *reinterpret_cast<const device float4*>(A_row_base);
        *reinterpret_cast<threadgroup float4*>(&Bs[0][bs_off]) =
            *reinterpret_cast<const device float4*>(B_row_base);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_k_tiles = K / BK;     // 256
    uint buf = 0;

    for (uint kt = 0; kt < num_k_tiles - 1u; ++kt) {
        const uint next = 1u - buf;

        // Prefetch next K-tile into 'next' buffer (overlaps with MMA below).
        *reinterpret_cast<threadgroup float4*>(&As[next][as_off]) =
            *reinterpret_cast<const device float4*>(A_row_base + (kt + 1u) * BK);
        *reinterpret_cast<threadgroup float4*>(&Bs[next][bs_off]) =
            *reinterpret_cast<const device float4*>(B_row_base + (kt + 1u) * BK * N);

        // Compute on current 'buf'.
        #pragma unroll
        for (uint kc = 0; kc < BK; kc += 8u) {
            simdgroup_matrix<float, 8, 8> A_blk[MMA_M], B_blk[MMA_N];
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                simdgroup_load(A_blk[i], &As[buf][(sm * SM + i * 8u) * LDA + kc], LDA);
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_load(B_blk[j], &Bs[buf][kc * LDB + sn * SN + j * 8u], LDB);
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                #pragma unroll
                for (uint j = 0; j < MMA_N; ++j)
                    simdgroup_multiply_accumulate(C_acc[i][j], A_blk[i], B_blk[j], C_acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf = next;
    }

    // Final tile.
    #pragma unroll
    for (uint kc = 0; kc < BK; kc += 8u) {
        simdgroup_matrix<float, 8, 8> A_blk[MMA_M], B_blk[MMA_N];
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            simdgroup_load(A_blk[i], &As[buf][(sm * SM + i * 8u) * LDA + kc], LDA);
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            simdgroup_load(B_blk[j], &Bs[buf][kc * LDB + sn * SN + j * 8u], LDB);
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_multiply_accumulate(C_acc[i][j], A_blk[i], B_blk[j], C_acc[i][j]);
    }

    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            simdgroup_store(C_acc[i][j],
                            &C[(c_row0 + sm * SM + i * 8u) * N + (c_col0 + sn * SN + j * 8u)],
                            N);
}
