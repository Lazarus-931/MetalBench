// TEMPLATE.metal — copy this to src/kernels/<set>/<name>.metal as a starting point.
//
// Replace:
//   <metal_function>  — kernel function name (must match registry)
//   <BM>, <BN>, <BK>  — tile dimensions
//   <SM>, <SN>        — simdgroup dimensions
//
// Bindings come from the registry entry:
//   buffer(N) = input/output/scalar according to input_bindings + outputs + scalars

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// --- Tile constants ---------------------------------------------------------
constant constexpr uint BM = 64;    // output tile rows
constant constexpr uint BN = 64;    // output tile cols
constant constexpr uint BK = 16;    // K-tile size (16 or 32 typical)

// --- Simdgroup layout -------------------------------------------------------
constant constexpr uint SM = 16;    // rows per simdgroup (BM / simdgroups_m)
constant constexpr uint SN = 32;    // cols per simdgroup (BN / simdgroups_n)
constant constexpr uint SIMDS_N = BN / SN;
constant constexpr uint MMA_M = SM / 8;   // 8x8 accumulators along M
constant constexpr uint MMA_N = SN / 8;   // 8x8 accumulators along N

// --- Padding (avoid bank conflicts) ----------------------------------------
constant constexpr uint PAD = 4;
constant constexpr uint LDA = BK + PAD;   // padded stride for A tile
constant constexpr uint LDB = BN + PAD;   // padded stride for B tile

// --- Kernel entry point -----------------------------------------------------
kernel void <metal_function>(
    // Inputs (buffer indices from registry input_bindings)
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    // Output (buffer index from registry outputs)
    device       float* C   [[buffer(2)]],
    // Scalars (buffer indices + values from registry scalars)
    constant     uint&  M   [[buffer(3)]],
    constant     uint&  N   [[buffer(4)]],
    constant     uint&  K   [[buffer(5)]],
    // Thread identification
    uint2 tgid              [[threadgroup_position_in_grid]],
    uint  sgid              [[simdgroup_index_in_threadgroup]],
    uint  lid               [[thread_index_in_threadgroup]])
{
    // --- Threadgroup memory (double-buffered) -------------------------------
    threadgroup float As[2][BM * LDA];
    threadgroup float Bs[2][BK * LDB];

    // --- Thread mapping -----------------------------------------------------
    const uint sm = sgid / SIMDS_N;        // simdgroup row (0..SIMDS_M-1)
    const uint sn = sgid % SIMDS_N;        // simdgroup col (0..SIMDS_N-1)
    const uint c_row0 = tgid.y * BM;       // output tile top-left row
    const uint c_col0 = tgid.x * BN;       // output tile top-left col

    // A load: lid/4 -> row (0..BM-1), lid%4 -> column group (0..BK/4-1)
    const uint a_row = lid / 4;
    const uint a_c4  = lid % 4;
    // B load: stride adjustment
    const uint b_row = lid / 16;
    const uint b_c4  = lid % 16;

    // --- Accumulators -------------------------------------------------------
    simdgroup_matrix<float, 8, 8> C_acc[MMA_M][MMA_N];
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            C_acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    // --- Prologue: load first K-tile ----------------------------------------
    {
        // Load A tile (BM × BK) into As[0] at thread's position
        *reinterpret_cast<threadgroup float4*>(&As[0][a_row * LDA + a_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&A[(c_row0 + a_row) * K + a_c4 * 4]);
        // Load B tile (BK × BN) into Bs[0]
        *reinterpret_cast<threadgroup float4*>(&Bs[0][b_row * LDB + b_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&B[b_row * N + c_col0 + b_c4 * 4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Main loop ----------------------------------------------------------
    const uint num_k_tiles = K / BK;
    uint buf = 0;

    for (uint kt = 0; kt < num_k_tiles - 1; ++kt) {
        const uint next   = 1 - buf;
        const uint k0_nxt = (kt + 1) * BK;

        // Load next tile into As[next], Bs[next] while computing on As[buf], Bs[buf]
        *reinterpret_cast<threadgroup float4*>(&As[next][a_row * LDA + a_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&A[(c_row0 + a_row) * K + k0_nxt + a_c4 * 4]);
        *reinterpret_cast<threadgroup float4*>(&Bs[next][b_row * LDB + b_c4 * 4]) =
            *reinterpret_cast<const device float4*>(&B[(k0_nxt + b_row) * N + c_col0 + b_c4 * 4]);

        // MMA: iterate over K-chunks (BK/kFragSize iterations, kFragSize=8)
        #pragma unroll
        for (uint kc = 0; kc < BK; kc += 8) {
            simdgroup_matrix<float, 8, 8> A_blk[MMA_M], B_blk[MMA_N];
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                simdgroup_load(A_blk[i], &As[buf][(sm * SM + i * 8) * LDA + kc], LDA);
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_load(B_blk[j], &Bs[buf][kc * LDB + sn * SN + j * 8], LDB);
            #pragma unroll
            for (uint i = 0; i < MMA_M; ++i)
                #pragma unroll
                for (uint j = 0; j < MMA_N; ++j)
                    simdgroup_multiply_accumulate(C_acc[i][j], A_blk[i], B_blk[j], C_acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf = next;
    }

    // --- Epilogue: compute final K-tile -------------------------------------
    #pragma unroll
    for (uint kc = 0; kc < BK; kc += 8) {
        simdgroup_matrix<float, 8, 8> A_blk[MMA_M], B_blk[MMA_N];
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            simdgroup_load(A_blk[i], &As[buf][(sm * SM + i * 8) * LDA + kc], LDA);
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            simdgroup_load(B_blk[j], &Bs[buf][kc * LDB + sn * SN + j * 8], LDB);
        #pragma unroll
        for (uint i = 0; i < MMA_M; ++i)
            #pragma unroll
            for (uint j = 0; j < MMA_N; ++j)
                simdgroup_multiply_accumulate(C_acc[i][j], A_blk[i], B_blk[j], C_acc[i][j]);
    }

    // --- Store results ------------------------------------------------------
    #pragma unroll
    for (uint i = 0; i < MMA_M; ++i)
        #pragma unroll
        for (uint j = 0; j < MMA_N; ++j)
            simdgroup_store(C_acc[i][j],
                            &C[(c_row0 + sm * SM + i * 8) * N + (c_col0 + sn * SN + j * 8)],
                            N);
}
