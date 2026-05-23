// gelu_linear: y = gelu(x @ W). Fused matmul + erf-GELU in store phase.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// erf via 6th-order rational approximation (Abramowitz & Stegun 7.1.26), |err| < 1.5e-7.
// Vectorized to evaluate 4 lanes in parallel — saves scalar tail vs the original loop.
static inline float4 erf4(float4 z) {
    float4 az = fabs(z);
    float4 t  = 1.0f / (1.0f + 0.3275911f * az);
    float4 p  = ((((1.061405429f * t - 1.453152027f) * t
              + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    float4 y  = 1.0f - p * exp(-z * z);
    return copysign(y, z);
}
static inline float4 gelu4(float4 v) {
    const float k = 0.70710678118654752f; // 1/sqrt(2)
    return 0.5f * v * (1.0f + erf4(v * k));
}

constant constexpr uint BM  = 64, BN  = 64, BK  = 16;
constant constexpr uint SM  = 16, SN  = 32;
constant constexpr uint SIMDS_N = BN / SN;
constant constexpr uint MMA_M = SM / 8, MMA_N = SN / 8;
constant constexpr uint PAD = 4, LDA = BK + PAD, LDB = BN + PAD;

kernel void gelu_linear_f32(
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
    threadgroup float shared[2 * BM * LDA + 2 * BK * LDB];
    threadgroup float (*As)[BM * LDA] = (threadgroup float (*)[BM * LDA]) &shared[0];
    threadgroup float (*Bs)[BK * LDB] = (threadgroup float (*)[BK * LDB]) &shared[2 * BM * LDA];
    const uint sm = sgid / SIMDS_N, sn = sgid % SIMDS_N;
    const uint c_row0 = tgid.y * BM, c_col0 = tgid.x * BN;
    const uint a_row = lid / 4, a_c4 = lid % 4, b_row = lid / 16, b_c4 = lid % 16;

    simdgroup_matrix<float, 8, 8> C_acc[MMA_M][MMA_N];
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            C_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    { // Prologue
        *reinterpret_cast<threadgroup float4*>(&As[0][a_row * LDA + a_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&A[(c_row0 + a_row) * K + a_c4 * 4]);
        *reinterpret_cast<threadgroup float4*>(&Bs[0][b_row * LDB + b_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&B[b_row * N + c_col0 + b_c4 * 4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint num_k_tiles = K / BK; uint buf = 0;

    for (uint kt = 0; kt < num_k_tiles - 1; ++kt) {
        const uint next = 1 - buf, k0_nxt = (kt + 1) * BK;
        *reinterpret_cast<threadgroup float4*>(&As[next][a_row * LDA + a_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&A[(c_row0 + a_row) * K + k0_nxt + a_c4 * 4]);
        *reinterpret_cast<threadgroup float4*>(&Bs[next][b_row * LDB + b_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&B[(k0_nxt + b_row) * N + c_col0 + b_c4 * 4]);
        #pragma unroll
        for (uint kc = 0; kc < BK; kc += 8) {
            simdgroup_matrix<float, 8, 8> A_blk[MMA_M], B_blk[MMA_N];
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i) simdgroup_load(A_blk[i], &As[buf][(sm*SM+i*8)*LDA+kc], LDA);
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j) simdgroup_load(B_blk[j], &Bs[buf][kc*LDB+sn*SN+j*8], LDB);
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                #pragma unroll
                for (uint j = 0; j < MMA_N; ++j)
                    simdgroup_multiply_accumulate(C_acc[i][j], A_blk[i], B_blk[j], C_acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); buf = next;
    }
    #pragma unroll
    for (uint kc = 0; kc < BK; kc += 8) {
        simdgroup_matrix<float, 8, 8> A_blk[MMA_M], B_blk[MMA_N];
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i) simdgroup_load(A_blk[i], &As[buf][(sm*SM+i*8)*LDA+kc], LDA);
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j) simdgroup_load(B_blk[j], &Bs[buf][kc*LDB+sn*SN+j*8], LDB);
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_multiply_accumulate(C_acc[i][j], A_blk[i], B_blk[j], C_acc[i][j]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* Cs = &shared[0];
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            simdgroup_store(C_acc[i][j], &Cs[(sm*SM + i*8) * BN + sn*SN + j*8], BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_f4 = BM * BN / 4;
    #pragma unroll
    for (uint idx = lid; idx < total_f4; idx += 256) {
        uint r = (idx * 4) / BN;
        uint c = (idx * 4) % BN;
        float4 v = *reinterpret_cast<threadgroup float4*>(&Cs[r * BN + c]);
        *reinterpret_cast<device float4*>(&C[(c_row0 + r) * N + (c_col0 + c)]) = gelu4(v);
    }
}
