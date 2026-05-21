// fused_qkv_projection: y = x @ W. (M, K) @ (K, N).
// M=128, N=192, K=512. Tiled: each TG (256 threads) computes a BM x BN output block.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint BM = 16;
constant constexpr uint BN = 16;
constant constexpr uint BK = 16;
constant constexpr uint TG_SIZE = 256;
constant constexpr uint TILES_N = 192 / BN;   // 12

kernel void fused_qkv_projection_f32(
    device const float*  X        [[buffer(0)]],
    device const float*  W        [[buffer(1)]],
    device       float*  Y        [[buffer(2)]],
    constant     uint&   M        [[buffer(3)]],   // 128
    constant     uint&   N        [[buffer(4)]],   // 192
    constant     uint&   K        [[buffer(5)]],   // 512
    uint  tgid_lin               [[threadgroup_position_in_grid]],
    uint  lid                    [[thread_index_in_threadgroup]])
{
    // Map linear tg to (tile_m, tile_n). 8 x 12 = 96 tiles.
    const uint tm = tgid_lin / TILES_N;
    const uint tn = tgid_lin % TILES_N;
    if (tm * BM >= M) return;

    const uint row_block = tm * BM;
    const uint col_block = tn * BN;

    // Each thread owns one output (lm, ln) in the BM x BN tile.
    const uint lm = lid / BN;   // 0..15
    const uint ln = lid % BN;   // 0..15

    threadgroup float Xs[BM * BK];   // 256 floats
    threadgroup float Ws[BK * BN];   // 256 floats

    float acc = 0.0f;
    const uint num_k = K / BK;  // 32
    for (uint kt = 0; kt < num_k; ++kt) {
        const uint k0 = kt * BK;
        // Cooperative load: 256 threads, 256 floats each tile. 1 element/thread.
        // Xs[lm, kc]  <- X[row_block+lm, k0+kc]  with idx = lm*BK + kc, lid = lm*BN + ln, BN=BK=16.
        Xs[lid] = X[(row_block + lm) * K + k0 + ln];   // ln serves as kc
        // Ws[kc, ln] <- W[k0+kc, col_block+ln]; idx = lm*BN+ln corresponds to kc=lm, ln=ln
        Ws[lid] = W[(k0 + lm) * N + col_block + ln];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kc = 0; kc < BK; ++kc) {
            acc += Xs[lm * BK + kc] * Ws[kc * BN + ln];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    Y[(row_block + lm) * N + (col_block + ln)] = acc;
}
