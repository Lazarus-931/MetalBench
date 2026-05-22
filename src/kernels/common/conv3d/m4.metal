// conv3d: NDHWC tiled with threadgroup-memory input cache.
// x (N,D,H,W,C_in), w (C_out,R,R,R,C_in). N=4 D=H=W=32 C=32 K=64 R=3 stride=1 → out (4,30,30,30,64).
//
// Tile: each threadgroup processes one (n, d2, h2, w2_block) where w2_block covers
// 16 contiguous w2 positions → 16 w2 × 64 K = 1024 outputs per TG (exactly 1024 threads).
// We cache the strip x[n, d2:d2+3, h2:h2+3, w_start:w_start+18, :] in threadgroup memory.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint TILE_W = 16;
constant constexpr uint TILE_K = 64;
constant constexpr uint STRIP_W = TILE_W + 2;            // 18
constant constexpr uint STRIP_ELEMS = 3u * 3u * STRIP_W * 32u;  // 5184

kernel void conv3d_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  w       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   N       [[buffer(3)]],
    constant     uint&   C       [[buffer(4)]],
    constant     uint&   D       [[buffer(5)]],
    constant     uint&   H       [[buffer(6)]],
    constant     uint&   W       [[buffer(7)]],
    constant     uint&   K       [[buffer(8)]],
    constant     uint&   R       [[buffer(9)]],
    constant     uint&   stride  [[buffer(10)]],
    uint tid_in_tg [[thread_position_in_threadgroup]],
    uint tg_id     [[threadgroup_position_in_grid]])
{
    const uint D2 = (D - R) / stride + 1;
    const uint H2 = (H - R) / stride + 1;
    const uint W2 = (W - R) / stride + 1;
    const uint NUM_W_TILES = (W2 + TILE_W - 1) / TILE_W;
    const uint TILES_PER_N = D2 * H2 * NUM_W_TILES;
    const uint TOTAL_TILES = N * TILES_PER_N;
    const uint NUM_TGS = 64;

    threadgroup float xstrip[STRIP_ELEMS];

    const uint t_w = tid_in_tg / TILE_K;
    const uint t_k = tid_in_tg % TILE_K;

    for (uint tile = tg_id; tile < TOTAL_TILES; tile += NUM_TGS) {
        uint q = tile;
        uint n  = q / TILES_PER_N;             q %= TILES_PER_N;
        uint d2 = q / (H2 * NUM_W_TILES);      q %= (H2 * NUM_W_TILES);
        uint h2 = q / NUM_W_TILES;
        uint wt = q % NUM_W_TILES;
        uint w2_base = wt * TILE_W;
        uint w2_count = min(TILE_W, W2 - w2_base);
        uint w_window = min(STRIP_W, W - w2_base);

        for (uint i = tid_in_tg; i < STRIP_ELEMS; i += 1024) {
            uint qi = i;
            uint rd = qi / (3u * STRIP_W * 32u); qi %= (3u * STRIP_W * 32u);
            uint rh = qi / (STRIP_W * 32u);      qi %= (STRIP_W * 32u);
            uint rw = qi / 32u;
            uint c  = qi % 32u;
            float v = 0.0f;
            if (rw < w_window) {
                uint dx = d2 + rd;
                uint hx = h2 + rh;
                uint wx = w2_base + rw;
                v = x[(((n * D + dx) * H + hx) * W + wx) * C + c];
            }
            xstrip[i] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (t_w < w2_count) {
            float sum = 0.0f;
            uint w_off = t_k * 27u * 32u;
            #pragma clang loop unroll(full)
            for (uint rd = 0; rd < 3u; ++rd) {
                #pragma clang loop unroll(full)
                for (uint rh = 0; rh < 3u; ++rh) {
                    uint strip_base = ((rd * 3u + rh) * STRIP_W) * 32u;
                    #pragma clang loop unroll(full)
                    for (uint rw = 0; rw < 3u; ++rw) {
                        uint xbase = strip_base + (t_w + rw) * 32u;
                        device const float4* wv = (device const float4*)(w + w_off);
                        threadgroup const float4* xv = (threadgroup const float4*)(xstrip + xbase);
                        #pragma clang loop unroll(full)
                        for (uint cc = 0; cc < 8u; ++cc) {
                            float4 xv4 = xv[cc];
                            float4 wv4 = wv[cc];
                            sum += xv4.x * wv4.x + xv4.y * wv4.y + xv4.z * wv4.z + xv4.w * wv4.w;
                        }
                        w_off += 32u;
                    }
                }
            }
            uint w2 = w2_base + t_w;
            y[(((n * D2 + d2) * H2 + h2) * W2 + w2) * K + t_k] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
