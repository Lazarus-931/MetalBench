// rms_norm_linear: y = rms_norm(x) @ W. RMS norm fused into matmul prologue.
// Normalizes the A tile in threadgroup memory before the first MMA.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM  = 64, BN  = 64, BK  = 16;
constant constexpr uint SM  = 16, SN  = 32;
constant constexpr uint SIMDS_N = BN / SN;
constant constexpr uint MMA_M = SM / 8, MMA_N = SN / 8;
constant constexpr uint PAD = 4, LDA = BK + PAD, LDB = BN + PAD;

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
    threadgroup float As[2][BM * LDA], Bs[2][BK * LDB];
    const uint sm = sgid / SIMDS_N, sn = sgid % SIMDS_N;
    const uint c_row0 = tgid.y * BM, c_col0 = tgid.x * BN;
    const uint a_row = lid / 4, a_c4 = lid % 4, b_row = lid / 16, b_c4 = lid % 16;

    simdgroup_matrix<float, 8, 8> C_acc[MMA_M][MMA_N];
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            C_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    // Prologue: load A tile + compute RMS norm per row + normalize in-place
    {
        float4 av = *reinterpret_cast<const device float4*>(&A[(c_row0 + a_row) * K + a_c4 * 4]);
        *(reinterpret_cast<threadgroup float4*>(&As[0][a_row * LDA + a_c4 * 4])) = av;
        *reinterpret_cast<threadgroup float4*>(&Bs[0][b_row * LDB + b_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&B[b_row * N + c_col0 + b_c4 * 4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // RMS norm: compute per-row sumsq of the loaded A tile
    float sq = 0.0f;
    for (uint k = 0; k < BK; k++) {
        float v = As[0][a_row * LDA + k];
        sq += v * v;
    }
    float sumsq = simd_sum(sq);
    threadgroup float tg_sumsq[32];
    uint sg = lid >> 5;
    if ((lid & 31) == 0) tg_sumsq[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 32) {
        sumsq = tg_sumsq[lid];
        for (uint s = 16; s > 0; s >>= 1)
            sumsq += simd_shuffle_down(sumsq, s);
    }
    if (lid == 0) tg_sumsq[0] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = rsqrt(tg_sumsq[0] / float(K) + eps);

    // Normalize the A tile in threadgroup memory
    for (uint k = 0; k < BK; k++)
        As[0][a_row * LDA + k] *= inv_rms;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main matmul loop (same as rect_mm)
    const uint num_k_tiles = K / BK;
    uint buf = 0;
    for (uint kt = 0; kt < num_k_tiles - 1; ++kt) {
        const uint next = 1 - buf, k0_nxt = (kt + 1) * BK;

        // Load + normalize next A tile
        float4 av = *reinterpret_cast<const device float4*>(&A[(c_row0 + a_row) * K + k0_nxt + a_c4 * 4]);
        *(reinterpret_cast<threadgroup float4*>(&As[next][a_row * LDA + a_c4 * 4])) = av;
        *reinterpret_cast<threadgroup float4*>(&Bs[next][b_row * LDB + b_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&B[(k0_nxt + b_row) * N + c_col0 + b_c4 * 4]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // RMS norm the new A tile
        sq = 0.0f;
        for (uint k = 0; k < BK; k++) sq += As[next][a_row * LDA + k] * As[next][a_row * LDA + k];
        float sumsq2 = simd_sum(sq);
        if ((lid & 31) == 0) tg_sumsq[sg] = sumsq2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < 32) {
            sumsq2 = tg_sumsq[lid];
            for (uint s = 16; s > 0; s >>= 1) sumsq2 += simd_shuffle_down(sumsq2, s);
        }
        if (lid == 0) tg_sumsq[0] = sumsq2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_rms2 = rsqrt(tg_sumsq[0] / float(K) + eps);
        for (uint k = 0; k < BK; k++) As[next][a_row * LDA + k] *= inv_rms2;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA on current (already normalized) buf
        #pragma unroll
        for (uint kc = 0; kc < BK; kc += 8) {
            simdgroup_matrix<float, 8, 8> A_blk[MMA_M], B_blk[MMA_N];
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i) simdgroup_load(A_blk[i], &As[buf][(sm * SM + i * 8) * LDA + kc], LDA);
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j) simdgroup_load(B_blk[j], &Bs[buf][kc * LDB + sn * SN + j * 8], LDB);
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                #pragma unroll
                for (uint j = 0; j < MMA_N; ++j)
                    simdgroup_multiply_accumulate(C_acc[i][j], A_blk[i], B_blk[j], C_acc[i][j]);
        }
        buf = next;
    }

    // Epilogue MMA on final normalized buf
    #pragma unroll
    for (uint kc = 0; kc < BK; kc += 8) {
        simdgroup_matrix<float, 8, 8> A_blk[MMA_M], B_blk[MMA_N];
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i) simdgroup_load(A_blk[i], &As[buf][(sm * SM + i * 8) * LDA + kc], LDA);
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j) simdgroup_load(B_blk[j], &Bs[buf][kc * LDB + sn * SN + j * 8], LDB);
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
            simdgroup_store(C_acc[i][j], &C[(c_row0 + sm * SM + i * 8) * N + (c_col0 + sn * SN + j * 8)], N);
}
