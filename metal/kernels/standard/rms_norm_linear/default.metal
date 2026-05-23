// rms_norm_linear: y = rms_norm(x) @ W. Fused single-pass: sumsq accumulated
// during the matmul K-loop (sumsq applied as inv_rms scale at epilogue since
// matmul is linear in A). B is loaded contiguously into N-major TG layout and
// read transposed by simdgroup_load to avoid scatter-stores.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM  = 64, BN  = 64, BK  = 16;
constant constexpr uint SM  = 16, SN  = 32, SIMDS_N = BN / SN;
constant constexpr uint MMA_M = SM / 8, MMA_N = SN / 8;
constant constexpr uint PAD = 4;
constant constexpr uint LDA = BK + PAD;   // As: BM × LDA  (K-major)
constant constexpr uint LDB = BK + PAD;   // Bs: BN × LDB  (N-major, K-fast)
constant constexpr uint LDC = BN + PAD;

constant constexpr uint AS0 = 0;
constant constexpr uint AS1 = AS0 + BM * LDA;
constant constexpr uint BS0 = AS1 + BM * LDA;
constant constexpr uint BS1 = BS0 + BN * LDB;
constant constexpr uint SCRATCH = BS1 + BN * LDB;  // 2*64*20 + 2*64*20 = 5120 floats

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
    threadgroup float S[SCRATCH];
    threadgroup float inv_rms_row[BM];

    threadgroup float* Asb[2] = {S + AS0, S + AS1};
    threadgroup float* Bsb[2] = {S + BS0, S + BS1};

    const uint sm = sgid / SIMDS_N, sn = sgid % SIMDS_N;
    const uint c_row0 = tgid.y * BM, c_col0 = tgid.x * BN;
    const uint a_row = lid / 4, a_c4 = lid % 4;
    // For B: 256 threads, BN=64 rows × BK=16 cols = 1024 floats per tile = 256 float4
    // → 1 float4 per thread, contiguous along K.
    const uint b_n = lid / 4, b_k4 = lid % 4;

    float sumsq = 0.0f;

    // Initial tile preload (kt=0)
    {
        float4 va = *reinterpret_cast<const device float4*>(&A[(c_row0 + a_row) * K + a_c4 * 4]);
        sumsq += va.x*va.x + va.y*va.y + va.z*va.z + va.w*va.w;
        *reinterpret_cast<threadgroup float4*>(&Asb[0][a_row*LDA + a_c4*4]) = va;

        float4 vb = *reinterpret_cast<const device float4*>(&B[(c_col0 + b_n) * K + b_k4 * 4]);
        *reinterpret_cast<threadgroup float4*>(&Bsb[0][b_n * LDB + b_k4 * 4]) = vb;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> C_acc[MMA_M][MMA_N];
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            C_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    const uint nkt = K / BK; uint buf = 0;

    for (uint kt = 0; kt < nkt - 1; ++kt) {
        const uint next = 1 - buf, k0 = (kt + 1) * BK;
        float4 va = *reinterpret_cast<const device float4*>(&A[(c_row0 + a_row)*K + k0 + a_c4*4]);
        sumsq += va.x*va.x + va.y*va.y + va.z*va.z + va.w*va.w;
        *reinterpret_cast<threadgroup float4*>(&Asb[next][a_row*LDA + a_c4*4]) = va;

        float4 vb = *reinterpret_cast<const device float4*>(&B[(c_col0 + b_n) * K + k0 + b_k4 * 4]);
        *reinterpret_cast<threadgroup float4*>(&Bsb[next][b_n * LDB + b_k4 * 4]) = vb;

        #pragma unroll
        for (uint kc = 0; kc < BK; kc += 8) {
            simdgroup_matrix<float, 8, 8> Ab[MMA_M], Bb[MMA_N];
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                simdgroup_load(Ab[i], &Asb[buf][(sm*SM+i*8)*LDA + kc], LDA);
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                // Bs is N-major (n,k). simdgroup_load with transpose reads an 8x8 (k,n) tile.
                simdgroup_load(Bb[j], &Bsb[buf][(sn*SN + j*8) * LDB + kc], LDB, ulong2(0,0), true);
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
        for (uint i = 0; i < MMA_M; ++i)
            simdgroup_load(Ab[i], &Asb[buf][(sm*SM+i*8)*LDA + kc], LDA);
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            simdgroup_load(Bb[j], &Bsb[buf][(sn*SN + j*8) * LDB + kc], LDB, ulong2(0,0), true);
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_multiply_accumulate(C_acc[i][j], Ab[i], Bb[j], C_acc[i][j]);
    }

    // Reduce sumsq within each row's 4 threads
    sumsq += simd_shuffle_xor(sumsq, 1);
    sumsq += simd_shuffle_xor(sumsq, 2);
    if (a_c4 == 0) inv_rms_row[a_row] = rsqrt(sumsq / float(K) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Epilogue: store simdgroup tiles to shared scratch, scale by inv_rms, write to device.
    threadgroup float* Cs = S;  // BM*LDC = 64*68 = 4352 floats; SCRATCH=5120 ✓
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            simdgroup_store(C_acc[i][j], &Cs[(sm*SM + i*8) * LDC + sn*SN + j*8], LDC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint c_row_t = lid / 16;
    const uint c_c4_t  = lid % 16;
    #pragma unroll
    for (uint p = 0; p < 4; ++p) {
        uint row = c_row_t + p * 16;
        float scale = inv_rms_row[row];
        float4 v = *reinterpret_cast<threadgroup float4*>(&Cs[row * LDC + c_c4_t * 4]);
        v *= scale;
        *reinterpret_cast<device float4*>(&C[(c_row0 + row) * N + c_col0 + c_c4_t * 4]) = v;
    }
}
