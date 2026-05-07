// rms_norm_linear: y = rms_norm(x) @ W. Two-pass: sumsq reduction, then normalized matmul.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM  = 64, BN  = 64, BK  = 16;
constant constexpr uint SM  = 16, SN  = 32, SIMDS_N = BN / SN;
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

    // Pass 1: compute per-row sumsq for RMS norm
    float sq_sum = 0.0f;
    for (uint kt = 0; kt < K; kt += BK)
        for (uint k = kt + a_c4 * 4; k < kt + BK && k < K; k++)
            sq_sum += A[(c_row0 + a_row) * K + k] * A[(c_row0 + a_row) * K + k];

    float sumsq = simd_sum(sq_sum);
    threadgroup float tg_buf[32];
    uint sg = lid >> 5;
    if ((lid & 31) == 0) tg_buf[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 32) {
        sumsq = tg_buf[lid];
        for (uint s = 16; s > 0; s >>= 1) sumsq += simd_shuffle_down(sumsq, s);
    }
    if (lid == 0) tg_buf[0] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(tg_buf[0] / float(K) + eps);

    // Pass 2: matmul with normalized A
    simdgroup_matrix<float, 8, 8> C_acc[MMA_M][MMA_N];
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            C_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    {
        float4 v = *reinterpret_cast<const device float4*>(&A[(c_row0 + a_row) * K + a_c4 * 4]);
        *reinterpret_cast<threadgroup float4*>(&As[0][a_row*LDA + a_c4*4]) = inv_rms * v;
        *reinterpret_cast<threadgroup float4*>(&Bs[0][b_row*LDB + b_c4*4]) =
            *reinterpret_cast<const device float4*>(&B[b_row*N + c_col0 + b_c4*4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nkt = K / BK; uint buf = 0;

    for (uint kt = 0; kt < nkt - 1; ++kt) {
        const uint next = 1 - buf, k0 = (kt + 1) * BK;
        float4 v = *reinterpret_cast<const device float4*>(&A[(c_row0 + a_row)*K + k0 + a_c4*4]);
        *reinterpret_cast<threadgroup float4*>(&As[next][a_row*LDA + a_c4*4]) = inv_rms * v;
        *reinterpret_cast<threadgroup float4*>(&Bs[next][b_row*LDB + b_c4*4]) =
            *reinterpret_cast<const device float4*>(&B[(k0 + b_row)*N + c_col0 + b_c4*4]);
        #pragma unroll
        for (uint kc = 0; kc < BK; kc += 8) {
            simdgroup_matrix<float, 8, 8> Ab[MMA_M], Bb[MMA_N];
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i) simdgroup_load(Ab[i], &As[buf][(sm*SM+i*8)*LDA+kc], LDA);
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j) simdgroup_load(Bb[j], &Bs[buf][kc*LDB+sn*SN+j*8], LDB);
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                #pragma unroll
                for (uint j = 0; j < MMA_N; ++j)
                    simdgroup_multiply_accumulate(C_acc[i][j], Ab[i], Bb[j], C_acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); buf = next;
    }
    #pragma unroll
    for (uint kc = 0; kc < BK; kc += 8) {
        simdgroup_matrix<float, 8, 8> Ab[MMA_M], Bb[MMA_N];
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i) simdgroup_load(Ab[i], &As[buf][(sm*SM+i*8)*LDA+kc], LDA);
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j) simdgroup_load(Bb[j], &Bs[buf][kc*LDB+sn*SN+j*8], LDB);
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_multiply_accumulate(C_acc[i][j], Ab[i], Bb[j], C_acc[i][j]);
    }

    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            simdgroup_store(C_acc[i][j], &C[(c_row0+sm*SM+i*8)*N+(c_col0+sn*SN+j*8)], N);
}
