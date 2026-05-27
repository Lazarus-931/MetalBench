// swiglu: Y = silu(X @ Wg) * (X @ Wu). M=N=K=256.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM      = 32;
constant constexpr uint BN      = 64;
constant constexpr uint BK      = 16;
constant constexpr uint SM      = 8;
constant constexpr uint SN      = 32;
constant constexpr uint SIMDS_M = BM / SM;        // 4
constant constexpr uint SIMDS_N = BN / SN;        // 2
constant constexpr uint MMA_M   = SM / 8;         // 1
constant constexpr uint MMA_N   = SN / 8;         // 4
constant constexpr uint PAD     = 4;
constant constexpr uint LDA     = BK + PAD;       // 20
constant constexpr uint LDB     = BN + PAD;       // 68
constant constexpr uint LDC     = BN + PAD;       // 68
constant constexpr uint TILE_COLS = 4;            // 256 / BN
constant constexpr uint NUM_TILES = 32;           // (256/BM)*(256/BN) = 8*4

kernel void swiglu_f32(
    device const float*  X       [[buffer(0)]],
    device const float*  Wg      [[buffer(1)]],
    device const float*  Wu      [[buffer(2)]],
    device       float*  Y       [[buffer(3)]],
    constant     uint&   M       [[buffer(4)]],
    constant     uint&   N       [[buffer(5)]],
    constant     uint&   K       [[buffer(6)]],
    uint  tgid                  [[threadgroup_position_in_grid]],
    uint  sgid                  [[simdgroup_index_in_threadgroup]],
    uint  lid                   [[thread_index_in_threadgroup]])
{
    if (tgid >= NUM_TILES) return;

    const uint tile_r = tgid / TILE_COLS;        // 0..7
    const uint tile_c = tgid % TILE_COLS;        // 0..3
    const uint c_row0 = tile_r * BM;
    const uint c_col0 = tile_c * BN;

    const uint sm = sgid / SIMDS_N;              // 0..3
    const uint sn = sgid % SIMDS_N;              // 0..1

    threadgroup float As [BM * LDA];        // 32*20=640
    threadgroup float Bgs[BK * LDB];        // 16*68=1088
    threadgroup float Bus[BK * LDB];        // 1088
    threadgroup float Gmem[BM * LDC];       // 32*68=2176
    threadgroup float Umem[BM * LDC];       // 2176  total ~7168 floats = 28672 B

    simdgroup_matrix<float, 8, 8> Cg[MMA_M][MMA_N];
    simdgroup_matrix<float, 8, 8> Cu[MMA_M][MMA_N];
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j) {
            Cg[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
            Cu[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
        }

    const uint num_k_tiles = K / BK;        // 16
    const uint a_lid = lid;                 // for A: use lid<128
    const uint a_row = a_lid / 4;
    const uint a_c4  = a_lid % 4;
    const uint b_row = lid / 16;
    const uint b_c4  = lid % 16;

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        const uint k0 = kt * BK;

        if (lid < 128) {
            *reinterpret_cast<threadgroup float4*>(&As[a_row * LDA + a_c4 * 4]) =
                *reinterpret_cast<const device float4*>(&X[(c_row0 + a_row) * K + k0 + a_c4 * 4]);
        }
        *reinterpret_cast<threadgroup float4*>(&Bgs[b_row * LDB + b_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&Wg[(k0 + b_row) * N + c_col0 + b_c4 * 4]);
        *reinterpret_cast<threadgroup float4*>(&Bus[b_row * LDB + b_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&Wu[(k0 + b_row) * N + c_col0 + b_c4 * 4]);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kc = 0; kc < BK; kc += 8) {
            simdgroup_matrix<float, 8, 8> A_blk[MMA_M], Bg_blk[MMA_N], Bu_blk[MMA_N];
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                simdgroup_load(A_blk[i], &As[(sm * SM + i * 8) * LDA + kc], LDA);
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j) {
                simdgroup_load(Bg_blk[j], &Bgs[kc * LDB + sn * SN + j * 8], LDB);
                simdgroup_load(Bu_blk[j], &Bus[kc * LDB + sn * SN + j * 8], LDB);
            }
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                #pragma unroll
                for (uint j = 0; j < MMA_N; ++j) {
                    simdgroup_multiply_accumulate(Cg[i][j], A_blk[i], Bg_blk[j], Cg[i][j]);
                    simdgroup_multiply_accumulate(Cu[i][j], A_blk[i], Bu_blk[j], Cu[i][j]);
                }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j) {
            simdgroup_store(Cg[i][j], &Gmem[(sm * SM + i * 8) * LDC + sn * SN + j * 8], LDC);
            simdgroup_store(Cu[i][j], &Umem[(sm * SM + i * 8) * LDC + sn * SN + j * 8], LDC);
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint o_row_base = lid / 16;   // 0..15
    const uint o_c4       = lid % 16;
    #pragma unroll
    for (uint s = 0; s < 2; ++s) {
        const uint row = o_row_base + s * 16;
        float4 g4 = *reinterpret_cast<threadgroup float4*>(&Gmem[row * LDC + o_c4 * 4]);
        float4 u4 = *reinterpret_cast<threadgroup float4*>(&Umem[row * LDC + o_c4 * 4]);
        float4 out;
        out.x = (g4.x / (1.0f + fast::exp(-g4.x))) * u4.x;
        out.y = (g4.y / (1.0f + fast::exp(-g4.y))) * u4.y;
        out.z = (g4.z / (1.0f + fast::exp(-g4.z))) * u4.z;
        out.w = (g4.w / (1.0f + fast::exp(-g4.w))) * u4.w;
        *reinterpret_cast<device float4*>(&Y[(c_row0 + row) * N + c_col0 + o_c4 * 4]) = out;
    }
}
