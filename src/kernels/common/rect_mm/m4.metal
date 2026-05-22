// rect_matmul: C = A @ B (MxK @ KxN -> MxN f32). M=1024 N=2048 K=4096.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM  = 64;
constant constexpr uint BN  = 64;
constant constexpr uint BK  = 32;
constant constexpr uint SM  = 8;
constant constexpr uint SN  = 64;
constant constexpr uint SIMDS_N = BN / SN;
constant constexpr uint MMA_M   = SM / 8;
constant constexpr uint MMA_N   = SN / 8;
constant constexpr uint PAD = 4;
constant constexpr uint LDA = BK + PAD;   // 36
constant constexpr uint LDB = BN + PAD;   // 68

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
    threadgroup float As[BM * LDA];     // 64*36 = 2304 = 9216 B
    threadgroup float Bs[BK * LDB];     // 32*68 = 2176 = 8704 B  total 17920

    const uint sm = sgid / SIMDS_N;
    const uint sn = sgid % SIMDS_N;
    const uint c_row0 = tgid.y * BM;
    const uint c_col0 = tgid.x * BN;

    const uint a_row0 = lid / 8;          // 0..31
    const uint a_c4_0 = lid % 8;          // 0..7
    const uint b_row0 = lid / 16;         // 0..15
    const uint b_c4_0 = lid % 16;         // 0..15

    simdgroup_matrix<float, 8, 8> C_acc[MMA_M][MMA_N];
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            C_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    const uint num_k_tiles = K / BK;     // 4096/32 = 128

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        const uint k0 = kt * BK;

        *reinterpret_cast<threadgroup float4*>(&As[a_row0 * LDA + a_c4_0 * 4]) =
            *reinterpret_cast<const device float4*>(&A[(c_row0 + a_row0) * K + k0 + a_c4_0 * 4]);
        *reinterpret_cast<threadgroup float4*>(&As[(a_row0 + 32) * LDA + a_c4_0 * 4]) =
            *reinterpret_cast<const device float4*>(&A[(c_row0 + a_row0 + 32) * K + k0 + a_c4_0 * 4]);

        *reinterpret_cast<threadgroup float4*>(&Bs[b_row0 * LDB + b_c4_0 * 4]) =
            *reinterpret_cast<const device float4*>(&B[(k0 + b_row0) * N + c_col0 + b_c4_0 * 4]);
        *reinterpret_cast<threadgroup float4*>(&Bs[(b_row0 + 16) * LDB + b_c4_0 * 4]) =
            *reinterpret_cast<const device float4*>(&B[(k0 + b_row0 + 16) * N + c_col0 + b_c4_0 * 4]);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kc = 0; kc < BK; kc += 8) {
            simdgroup_matrix<float, 8, 8> A_blk[MMA_M], B_blk[MMA_N];
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                simdgroup_load(A_blk[i], &As[(sm * SM + i * 8) * LDA + kc], LDA);
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_load(B_blk[j], &Bs[kc * LDB + sn * SN + j * 8], LDB);
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
            simdgroup_store(C_acc[i][j],
                            &C[(c_row0 + sm * SM + i * 8) * N + (c_col0 + sn * SN + j * 8)],
                            N);
}
