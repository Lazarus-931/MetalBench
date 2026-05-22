// matmul_gelu_softmax: y = softmax(GELU(x @ w), axis=-1)
// x (256,256), w (256,256), output (256,256). Grid (256,256,1) TG (256,1,1).
// BM=16, BN=256, BK=8. 16 TGs across 10 M4 cores. Double-buffered B.
//
// 8 simdgroups (256 threads). SIMDS_M=2, SIMDS_N=4 (SM=8, SN=64). MMA_M=1, MMA_N=8.
// Softmax: 16 threads/row, simd_shuffle_xor stride 1..8 inside a 16-lane half.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM = 16;
constant constexpr uint BN = 256;
constant constexpr uint BK = 16;
constant constexpr uint LDA = BK;
constant constexpr uint LDB = BN;
constant constexpr uint LDC = BN;

constant constexpr uint SIMDS_M = 2;
constant constexpr uint SIMDS_N = 4;
constant constexpr uint SM = BM / SIMDS_M;  // 8
constant constexpr uint SN = BN / SIMDS_N;  // 64
constant constexpr uint MMA_N = SN / 8;     // 8

kernel void matmul_gelu_softmax_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device       float* Y [[buffer(2)]],
    constant     uint&  M [[buffer(3)]],
    constant     uint&  N [[buffer(4)]],
    constant     uint&  K [[buffer(5)]],
    uint3 tgid            [[threadgroup_position_in_grid]],
    uint  sgid            [[simdgroup_index_in_threadgroup]],
    uint  lid             [[thread_index_in_threadgroup]])
{
    const uint ty = tgid.y;
    if (ty >= 16) return;

    threadgroup float As[BM * LDA];        // 16*16*4 = 1024 B
    // Bs (BK*LDB=16*256=16384 B) and Cs (BM*LDC=16*256=16384 B) overlay the
    // same TG buffer: Bs lifetime ends before Cs begins (after MMA finishes).
    threadgroup float shared_bc[BM * LDC]; // 16384 B
    threadgroup float* Bs = shared_bc;     // alias during matmul
    threadgroup float* Cs = shared_bc;     // alias after matmul

    const uint sm = sgid / SIMDS_N;
    const uint sn = sgid % SIMDS_N;
    const uint c_row0 = ty * BM;

    simdgroup_matrix<float, 8, 8> C_acc[MMA_N];
    #pragma unroll
    for (uint j = 0; j < MMA_N; ++j)
        C_acc[j] = simdgroup_matrix<float, 8, 8>(0.0f);

    const uint num_k_tiles = K / BK;       // 32

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        const uint k0 = kt * BK;
        // A: 16 rows × 16 cols = 256 floats = 64 float4 → 64 threads.
        if (lid < 64) {
            uint ar = lid >> 2;
            uint ac4 = lid & 3;
            *reinterpret_cast<threadgroup float4*>(&As[ar * LDA + ac4 * 4]) =
                *reinterpret_cast<const device float4*>(&X[(c_row0 + ar) * K + k0 + ac4 * 4]);
        }
        // B: 16 rows × 256 cols = 1024 float4 → 4 per thread.
        #pragma unroll
        for (uint q = 0; q < 4; ++q) {
            uint idx = q * 256 + lid;
            uint br = idx / 64;
            uint bc4 = idx % 64;
            *reinterpret_cast<threadgroup float4*>(&Bs[br * LDB + bc4 * 4]) =
                *reinterpret_cast<const device float4*>(&W[(k0 + br) * N + bc4 * 4]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kc = 0; kc < BK; kc += 8) {
            simdgroup_matrix<float, 8, 8> A_blk, B_blk[MMA_N];
            simdgroup_load(A_blk, &As[(sm * SM) * LDA + kc], LDA);
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_load(B_blk[j], &Bs[kc * LDB + sn * SN + j * 8], LDB);
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_multiply_accumulate(C_acc[j], A_blk, B_blk[j], C_acc[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint j = 0; j < MMA_N; ++j)
        simdgroup_store(C_acc[j], &Cs[(sm * SM) * LDC + sn * SN + j * 8], LDC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Softmax phase (same as before)
    const uint r = lid >> 4;
    const uint lane16 = lid & 15;
    const uint col_base = lane16 * 16;

    const float k_inv_sqrt2 = 0.70710678118654752f;
    float vals[16];
    float local_max = -INFINITY;

    threadgroup const float* Crow = &Cs[r * LDC + col_base];
    #pragma unroll
    for (uint t = 0; t < 16; t += 4) {
        float4 v = *reinterpret_cast<threadgroup const float4*>(Crow + t);
        float ss[4] = { v.x, v.y, v.z, v.w };
        #pragma unroll
        for (uint u = 0; u < 4; ++u) {
            float s = ss[u];
            float z = s * k_inv_sqrt2;
            float at = 1.0f / (1.0f + 0.3275911f * fabs(z));
            float yp = 1.0f - (((((1.061405429f * at - 1.453152027f) * at)
                          + 1.421413741f) * at - 0.284496736f) * at + 0.254829592f) * at * precise::exp(-z * z);
            float erfz = copysign(yp, z);
            float gv = 0.5f * s * (1.0f + erfz);
            vals[t + u] = gv;
            local_max = fmax(local_max, gv);
        }
    }
    float rmax = local_max;
    rmax = fmax(rmax, simd_shuffle_xor(rmax, 1));
    rmax = fmax(rmax, simd_shuffle_xor(rmax, 2));
    rmax = fmax(rmax, simd_shuffle_xor(rmax, 4));
    rmax = fmax(rmax, simd_shuffle_xor(rmax, 8));

    float local_sum = 0.0f;
    #pragma unroll
    for (uint u = 0; u < 16; ++u) {
        float e = precise::exp(vals[u] - rmax);
        vals[u] = e;
        local_sum += e;
    }
    float rsum = local_sum;
    rsum += simd_shuffle_xor(rsum, 1);
    rsum += simd_shuffle_xor(rsum, 2);
    rsum += simd_shuffle_xor(rsum, 4);
    rsum += simd_shuffle_xor(rsum, 8);
    float inv = 1.0f / rsum;

    device float* Yrow = &Y[(c_row0 + r) * N + col_base];
    #pragma unroll
    for (uint t = 0; t < 16; t += 4) {
        float4 o = { vals[t]*inv, vals[t+1]*inv, vals[t+2]*inv, vals[t+3]*inv };
        *reinterpret_cast<device float4*>(Yrow + t) = o;
    }
}
