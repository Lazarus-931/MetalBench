// silu_linear: y = silu(x @ W). Fused matmul + SiLU in store phase.
// Tile: BM=64 BN=64 BK=32, double-buffered, 8 simdgroups, 256 threads.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM  = 64, BN  = 64, BK  = 32;
constant constexpr uint SM  = 16, SN  = 32;
constant constexpr uint SIMDS_N = BN / SN;          // 2
constant constexpr uint MMA_M = SM / 8, MMA_N = SN / 8;  // 2, 4
constant constexpr uint PAD = 0, LDA = BK + PAD, LDB = BN + PAD;  // 32, 64

kernel void silu_linear_f32(
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

    // A loader: 256 threads cover 64×32 floats = 512 f4. row=lid/8 (0..31), c4=lid%8 (0..7).
    // Each thread loads 2 rows: a_row and a_row+32.
    const uint a_row = lid / 8;
    const uint a_c4  = lid % 8;
    // B loader: 32×64 floats = 512 f4. row=lid/16 (0..15), c4=lid%16 (0..15).
    // Each thread loads 2 rows: b_row and b_row+16.
    const uint b_row = lid / 16;
    const uint b_c4  = lid % 16;

    simdgroup_matrix<float, 8, 8> C_acc[MMA_M][MMA_N];
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            C_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    // ---- Prologue: load tile 0 ----
    {
        const uint k0 = 0;
        // A: rows (c_row0 + a_row) and (c_row0 + a_row + 32), cols [k0 + a_c4*4 .. +3]
        *reinterpret_cast<threadgroup float4*>(&As[0][a_row * LDA + a_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&A[(c_row0 + a_row) * K + k0 + a_c4 * 4]);
        *reinterpret_cast<threadgroup float4*>(&As[0][(a_row + 32) * LDA + a_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&A[(c_row0 + a_row + 32) * K + k0 + a_c4 * 4]);
        // B: rows (k0 + b_row) and (k0 + b_row + 16), cols [c_col0 + b_c4*4 .. +3]
        *reinterpret_cast<threadgroup float4*>(&Bs[0][b_row * LDB + b_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&B[(k0 + b_row) * N + c_col0 + b_c4 * 4]);
        *reinterpret_cast<threadgroup float4*>(&Bs[0][(b_row + 16) * LDB + b_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&B[(k0 + b_row + 16) * N + c_col0 + b_c4 * 4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_k_tiles = K / BK;
    uint buf = 0;

    for (uint kt = 0; kt < num_k_tiles - 1; ++kt) {
        const uint next = 1 - buf;
        const uint k0_nxt = (kt + 1) * BK;

        // Issue next-tile loads
        *reinterpret_cast<threadgroup float4*>(&As[next][a_row * LDA + a_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&A[(c_row0 + a_row) * K + k0_nxt + a_c4 * 4]);
        *reinterpret_cast<threadgroup float4*>(&As[next][(a_row + 32) * LDA + a_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&A[(c_row0 + a_row + 32) * K + k0_nxt + a_c4 * 4]);
        *reinterpret_cast<threadgroup float4*>(&Bs[next][b_row * LDB + b_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&B[(k0_nxt + b_row) * N + c_col0 + b_c4 * 4]);
        *reinterpret_cast<threadgroup float4*>(&Bs[next][(b_row + 16) * LDB + b_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&B[(k0_nxt + b_row + 16) * N + c_col0 + b_c4 * 4]);

        // Compute over current tile
        #pragma unroll
        for (uint kc = 0; kc < BK; kc += 8) {
            simdgroup_matrix<float, 8, 8> A_blk[MMA_M], B_blk[MMA_N];
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                simdgroup_load(A_blk[i], &As[buf][(sm*SM + i*8)*LDA + kc], LDA);
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_load(B_blk[j], &Bs[buf][kc*LDB + sn*SN + j*8], LDB);
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                #pragma unroll
                for (uint j = 0; j < MMA_N; ++j)
                    simdgroup_multiply_accumulate(C_acc[i][j], A_blk[i], B_blk[j], C_acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf = next;
    }

    // Epilogue tile compute
    #pragma unroll
    for (uint kc = 0; kc < BK; kc += 8) {
        simdgroup_matrix<float, 8, 8> A_blk[MMA_M], B_blk[MMA_N];
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            simdgroup_load(A_blk[i], &As[buf][(sm*SM + i*8)*LDA + kc], LDA);
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            simdgroup_load(B_blk[j], &Bs[buf][kc*LDB + sn*SN + j*8], LDB);
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_multiply_accumulate(C_acc[i][j], A_blk[i], B_blk[j], C_acc[i][j]);
    }

    // Store C tiles to shared, then SiLU + write to global
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* Cs = &shared[0];
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            simdgroup_store(C_acc[i][j], &Cs[(sm*SM + i*8) * BN + sn*SN + j*8], BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_f4 = BM * BN / 4;  // 1024
    #pragma unroll
    for (uint idx = lid; idx < total_f4; idx += 256) {
        uint r = (idx * 4) / BN;
        uint c = (idx * 4) % BN;
        float4 v = *reinterpret_cast<threadgroup float4*>(&Cs[r * BN + c]);
        float4 out = v / (1.0f + exp(-v));
        *reinterpret_cast<device float4*>(&C[(c_row0 + r) * N + (c_col0 + c)]) = out;
    }
}
