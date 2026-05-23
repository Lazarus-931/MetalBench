// rms_norm_linear: y = rms_norm(x) @ W.T  where W is (N,K).
// M4 variant: BK=32 single-buffered with software-pipelined device prefetch.
// B is stored in shared in N-major (n, k) layout — read via transposed
// simdgroup_load to feed MMA as (k, n) tiles. sumsq accumulated during
// matmul K-loop (matmul linear in A → inv_rms applied at epilogue).
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM  = 64;
constant constexpr uint BN  = 64;
constant constexpr uint BK  = 32;
constant constexpr uint SM  = 16;
constant constexpr uint SN  = 32;
constant constexpr uint SIMDS_N = BN / SN;        // 2
constant constexpr uint MMA_M = SM / 8;           // 2
constant constexpr uint MMA_N = SN / 8;           // 4
constant constexpr uint PAD = 0;                  // M4: PAD=0 keeps TG mem low → better residency
constant constexpr uint LDA = BK + PAD;           // 32
constant constexpr uint LDB = BK + PAD;           // 32 (N-major, K-fast)

kernel void rms_norm_linear_f32(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device       float* C   [[buffer(2)]],
    constant     uint&  M   [[buffer(3)]],
    constant     uint&  N   [[buffer(4)]],
    constant     uint&  K   [[buffer(5)]],
    constant     float&  eps [[buffer(6)]],
    uint2 tgid              [[threadgroup_position_in_grid]],
    uint  sgid              [[simdgroup_index_in_threadgroup]],
    uint  lid               [[thread_index_in_threadgroup]])
{
    // Shared layout: As [BM x LDA] then Bs [BN x LDB] then Cs reuses start.
    constexpr uint AS_SZ = BM * LDA;     // 64*36 = 2304
    constexpr uint BS_SZ = BN * LDB;     // 64*36 = 2304
    constexpr uint CS_SZ = BM * (BN + 4);// 64*68 = 4352 (LDC=BN+4)
    constexpr uint LDC = BN + 4;
    constexpr uint SCRATCH = (AS_SZ + BS_SZ > CS_SZ) ? (AS_SZ + BS_SZ) : CS_SZ;
    threadgroup float S[SCRATCH];
    threadgroup float inv_rms_row[BM];

    threadgroup float* As = S;
    threadgroup float* Bs = S + AS_SZ;
    threadgroup float* Cs = S;

    const uint sm = sgid / SIMDS_N;
    const uint sn = sgid % SIMDS_N;
    const uint c_row0 = tgid.y * BM;
    const uint c_col0 = tgid.x * BN;

    // Loads: 256 threads, BM*BK = 2048 floats = 512 float4s → 2 float4/thread for A.
    //        Same for B (BN*BK = 2048).
    // For A: load 2 rows per thread, BK=32 → 8 float4 per row, so a_row_t = lid/8 selects
    //        row out of 32; we load row a_row_t and a_row_t+32. a_c8 = lid % 8 → col chunk.
    const uint a_row = lid / 8u;          // 0..31
    const uint a_c8  = lid % 8u;          // 0..7 → col4 base
    const uint b_n   = lid / 8u;          // 0..31  (N row idx in tile)
    const uint b_k8  = lid % 8u;          // 0..7 → k4 base

    const uint a_sh0 = a_row * LDA + a_c8 * 4u;
    const uint a_sh1 = (a_row + 32u) * LDA + a_c8 * 4u;
    const uint b_sh0 = b_n * LDB + b_k8 * 4u;
    const uint b_sh1 = (b_n + 32u) * LDB + b_k8 * 4u;
    const uint a_g0  = (c_row0 + a_row) * K + a_c8 * 4u;
    const uint a_g1  = (c_row0 + a_row + 32u) * K + a_c8 * 4u;
    const uint b_g0  = (c_col0 + b_n) * K + b_k8 * 4u;
    const uint b_g1  = (c_col0 + b_n + 32u) * K + b_k8 * 4u;

    float sumsq_lo = 0.0f;
    float sumsq_hi = 0.0f;

    // Prologue tile 0
    float4 a_lo = *reinterpret_cast<const device float4*>(&A[a_g0]);
    float4 a_hi = *reinterpret_cast<const device float4*>(&A[a_g1]);
    float4 b_lo = *reinterpret_cast<const device float4*>(&B[b_g0]);
    float4 b_hi = *reinterpret_cast<const device float4*>(&B[b_g1]);
    sumsq_lo += a_lo.x*a_lo.x + a_lo.y*a_lo.y + a_lo.z*a_lo.z + a_lo.w*a_lo.w;
    sumsq_hi += a_hi.x*a_hi.x + a_hi.y*a_hi.y + a_hi.z*a_hi.z + a_hi.w*a_hi.w;
    *reinterpret_cast<threadgroup float4*>(&As[a_sh0]) = a_lo;
    *reinterpret_cast<threadgroup float4*>(&As[a_sh1]) = a_hi;
    *reinterpret_cast<threadgroup float4*>(&Bs[b_sh0]) = b_lo;
    *reinterpret_cast<threadgroup float4*>(&Bs[b_sh1]) = b_hi;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> C_acc[MMA_M][MMA_N];
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            C_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    const uint num_k_tiles = K / BK;
    for (uint kt = 0; kt < num_k_tiles - 1u; ++kt) {
        const uint k0n = (kt + 1u) * BK;
        a_lo = *reinterpret_cast<const device float4*>(&A[a_g0 + k0n]);
        a_hi = *reinterpret_cast<const device float4*>(&A[a_g1 + k0n]);
        b_lo = *reinterpret_cast<const device float4*>(&B[b_g0 + k0n]);
        b_hi = *reinterpret_cast<const device float4*>(&B[b_g1 + k0n]);
        sumsq_lo += a_lo.x*a_lo.x + a_lo.y*a_lo.y + a_lo.z*a_lo.z + a_lo.w*a_lo.w;
        sumsq_hi += a_hi.x*a_hi.x + a_hi.y*a_hi.y + a_hi.z*a_hi.z + a_hi.w*a_hi.w;

        #pragma unroll
        for (uint kc = 0; kc < BK; kc += 8u) {
            simdgroup_matrix<float, 8, 8> Ab[MMA_M], Bb[MMA_N];
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                simdgroup_load(Ab[i], &As[(sm*SM + i*8u)*LDA + kc], LDA);
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_load(Bb[j], &Bs[(sn*SN + j*8u)*LDB + kc], LDB, ulong2(0,0), true);
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                #pragma unroll
                for (uint j = 0; j < MMA_N; ++j)
                    simdgroup_multiply_accumulate(C_acc[i][j], Ab[i], Bb[j], C_acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        *reinterpret_cast<threadgroup float4*>(&As[a_sh0]) = a_lo;
        *reinterpret_cast<threadgroup float4*>(&As[a_sh1]) = a_hi;
        *reinterpret_cast<threadgroup float4*>(&Bs[b_sh0]) = b_lo;
        *reinterpret_cast<threadgroup float4*>(&Bs[b_sh1]) = b_hi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Tail tile
    #pragma unroll
    for (uint kc = 0; kc < BK; kc += 8u) {
        simdgroup_matrix<float, 8, 8> Ab[MMA_M], Bb[MMA_N];
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            simdgroup_load(Ab[i], &As[(sm*SM + i*8u)*LDA + kc], LDA);
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            simdgroup_load(Bb[j], &Bs[(sn*SN + j*8u)*LDB + kc], LDB, ulong2(0,0), true);
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_multiply_accumulate(C_acc[i][j], Ab[i], Bb[j], C_acc[i][j]);
    }

    // Reduce sumsq across the 8 threads sharing a_row (a_c8 = 0..7).
    // Lanes layout: lid = a_row * 8 + a_c8 → 8 consecutive lanes share a_row.
    // Each simd has 32 lanes = 4 rows. XOR with 1, 2, 4 reduces within the 8-lane group.
    sumsq_lo += simd_shuffle_xor(sumsq_lo, 1);
    sumsq_lo += simd_shuffle_xor(sumsq_lo, 2);
    sumsq_lo += simd_shuffle_xor(sumsq_lo, 4);
    sumsq_hi += simd_shuffle_xor(sumsq_hi, 1);
    sumsq_hi += simd_shuffle_xor(sumsq_hi, 2);
    sumsq_hi += simd_shuffle_xor(sumsq_hi, 4);
    if (a_c8 == 0) {
        inv_rms_row[a_row]        = rsqrt(sumsq_lo / float(K) + eps);
        inv_rms_row[a_row + 32u]  = rsqrt(sumsq_hi / float(K) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Epilogue: store simdgroup tiles to shared, scale by inv_rms, write float4 to device.
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            simdgroup_store(C_acc[i][j], &Cs[(sm*SM + i*8u) * LDC + sn*SN + j*8u], LDC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint c_row_t = lid / 16u;
    const uint c_c4_t  = lid % 16u;
    #pragma unroll
    for (uint p = 0; p < 4u; ++p) {
        uint row = c_row_t + p * 16u;
        float scale = inv_rms_row[row];
        float4 v = *reinterpret_cast<threadgroup float4*>(&Cs[row * LDC + c_c4_t * 4u]);
        v *= scale;
        *reinterpret_cast<device float4*>(&C[(c_row0 + row) * N + c_col0 + c_c4_t * 4u]) = v;
    }
}
