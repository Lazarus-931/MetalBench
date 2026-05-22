// fused_qkv_projection — M4-tuned v2. Y = X @ W. (128,512) @ (512,192).
// M4 playbook:
//  * BM=16, BN=32 -> 48 active TGs (vs 6 default), saturates 10-core M4.
//  * BK=32, PAD=0 — wide K-tile, no bank padding (32-wide rows align cleanly).
//  * Direct simdgroup_store to device Y (no Cs round-trip).
//  * Register-pipelined prefetch via double-buffered threadgroup memory.
//  * float4 cooperative loads, hoisted index decomposition.
//  * 256-thread TG honored, 8 simdgroups (2x4) cover the 16x32 C tile via
//    8x8 simdgroup_matrix ops (MMA_M=1, MMA_N=1, each simd owns one 8x8).
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM = 16, BN = 32, BK = 32;
constant constexpr uint SM = 8,  SN = 8;
constant constexpr uint SIMDS_M = BM / SM;        // 2
constant constexpr uint SIMDS_N = BN / SN;        // 4
constant constexpr uint MMA_M   = SM / 8;         // 1
constant constexpr uint MMA_N   = SN / 8;         // 1
constant constexpr uint PAD     = 0;
constant constexpr uint LDA     = BK + PAD;       // 32
constant constexpr uint LDB     = BN + PAD;       // 32
constant constexpr uint TILES_N = 192 / BN;       // 6
constant constexpr uint TILES_M = 128 / BM;       // 8
constant constexpr uint ACTIVE_TG = TILES_M * TILES_N;  // 48

kernel void fused_qkv_projection_f32(
    device const float* X    [[buffer(0)]],
    device const float* W    [[buffer(1)]],
    device       float* Y    [[buffer(2)]],
    constant     uint&  M    [[buffer(3)]],
    constant     uint&  N    [[buffer(4)]],
    constant     uint&  K    [[buffer(5)]],
    uint tgid_lin            [[threadgroup_position_in_grid]],
    uint sgid                [[simdgroup_index_in_threadgroup]],
    uint lid                 [[thread_index_in_threadgroup]])
{
    if (tgid_lin >= ACTIVE_TG) return;
    const uint tx = tgid_lin % TILES_N;
    const uint ty = tgid_lin / TILES_N;

    threadgroup float As[2][BM * LDA];   // 2 * 16 * 32 = 1024 floats
    threadgroup float Bs[2][BK * LDB];   // 2 * 32 * 32 = 2048 floats  (total 12 KB)

    const uint sm = sgid / SIMDS_N;       // 0..1
    const uint sn = sgid % SIMDS_N;       // 0..3
    const uint c_row0 = ty * BM;
    const uint c_col0 = tx * BN;

    // A load: 16*32 = 512 floats = 128 float4. Use lower half of TG (lid<128).
    // B load: 32*32 = 1024 floats = 256 float4. Use full TG (one each).
    const uint a_row = lid >> 3;          // 0..31, but we only use lid<128 -> 0..15
    const uint a_c4  = lid & 7;
    const uint b_row = lid >> 3;          // 0..31
    const uint b_c4  = lid & 7;

    simdgroup_matrix<float, 8, 8> C_acc = simdgroup_matrix<float, 8, 8>(0.0f);

    // Prologue: load tile 0.
    if (lid < 128) {
        *reinterpret_cast<threadgroup float4*>(&As[0][a_row * LDA + a_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&X[(c_row0 + a_row) * K + a_c4 * 4]);
    }
    *reinterpret_cast<threadgroup float4*>(&Bs[0][b_row * LDB + b_c4 * 4]) =
        *reinterpret_cast<const device float4*>(&W[b_row * N + c_col0 + b_c4 * 4]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_k_tiles = K / BK;   // 16
    uint buf = 0;

    for (uint kt = 0; kt < num_k_tiles - 1; ++kt) {
        const uint next = 1 - buf;
        const uint k0_nxt = (kt + 1) * BK;

        if (lid < 128) {
            *reinterpret_cast<threadgroup float4*>(&As[next][a_row * LDA + a_c4 * 4]) =
                *reinterpret_cast<const device float4*>(&X[(c_row0 + a_row) * K + k0_nxt + a_c4 * 4]);
        }
        *reinterpret_cast<threadgroup float4*>(&Bs[next][b_row * LDB + b_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&W[(k0_nxt + b_row) * N + c_col0 + b_c4 * 4]);

        #pragma unroll
        for (uint kc = 0; kc < BK; kc += 8) {
            simdgroup_matrix<float, 8, 8> A_blk, B_blk;
            simdgroup_load(A_blk, &As[buf][(sm * SM) * LDA + kc], LDA);
            simdgroup_load(B_blk, &Bs[buf][kc * LDB + sn * SN], LDB);
            simdgroup_multiply_accumulate(C_acc, A_blk, B_blk, C_acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf = next;
    }
    #pragma unroll
    for (uint kc = 0; kc < BK; kc += 8) {
        simdgroup_matrix<float, 8, 8> A_blk, B_blk;
        simdgroup_load(A_blk, &As[buf][(sm * SM) * LDA + kc], LDA);
        simdgroup_load(B_blk, &Bs[buf][kc * LDB + sn * SN], LDB);
        simdgroup_multiply_accumulate(C_acc, A_blk, B_blk, C_acc);
    }

    simdgroup_store(C_acc, &Y[(c_row0 + sm * SM) * N + (c_col0 + sn * SN)], N);
}
