// conv_transpose2d: NHWC. x (N=8, H=32, W=32, C_in=64), w (C_out=128, R=3, S=3, C_in=64),
#include <metal_stdlib>
using namespace metal;

constant constexpr uint W_BLOCK = 8;
constant constexpr uint OUT_K  = 128;
constant constexpr uint C_IN   = 64;
constant constexpr uint R_K    = 3;

kernel void conv_transpose2d_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  w       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   N       [[buffer(3)]],
    constant     uint&   C_in    [[buffer(4)]],
    constant     uint&   H       [[buffer(5)]],
    constant     uint&   W       [[buffer(6)]],
    constant     uint&   C_out   [[buffer(7)]],
    constant     uint&   R       [[buffer(8)]],
    constant     uint&   stride  [[buffer(9)]],
    uint tid_in_tg [[thread_position_in_threadgroup]],
    uint tg_id     [[threadgroup_position_in_grid]])
{
    const uint H_out = (H - 1) * stride + R;          // 65
    const uint W_out = (W - 1) * stride + R;          // 65
    const uint NW_TILES = (W_out + W_BLOCK - 1) / W_BLOCK;  // ceil(65/8)=9
    const uint TILES = N * H_out * NW_TILES;          // 8*65*9 = 4680
    const uint NUM_TGS = 64;

    const uint t_w = tid_in_tg / OUT_K;        // 0..7
    const uint t_k = tid_in_tg % OUT_K;        // 0..127

    for (uint tile = tg_id; tile < TILES; tile += NUM_TGS) {
        uint q = tile;
        uint n     = q / (H_out * NW_TILES);   q %= (H_out * NW_TILES);
        uint h_out = q / NW_TILES;
        uint wt    = q % NW_TILES;
        uint w_out_base = wt * W_BLOCK;

        uint w_out = w_out_base + t_w;
        if (w_out >= W_out) continue;

        float sum = 0.0f;

        #pragma clang loop unroll(full)
        for (uint r = 0; r < R_K; ++r) {
            int h_in_s = int(h_out) - int(r);
            if (h_in_s < 0) continue;
            if ((h_in_s & 1) != 0) continue;          // stride=2
            uint h_in = uint(h_in_s) >> 1;
            if (h_in >= H) continue;

            #pragma clang loop unroll(full)
            for (uint s = 0; s < R_K; ++s) {
                int w_in_s = int(w_out) - int(s);
                if (w_in_s < 0) continue;
                if ((w_in_s & 1) != 0) continue;
                uint w_in = uint(w_in_s) >> 1;
                if (w_in >= W) continue;

                device const float4* xv = (device const float4*)(x + ((n * H + h_in) * W + w_in) * C_IN);
                device const float4* wv = (device const float4*)(w + ((t_k * R_K + r) * R_K + s) * C_IN);
                #pragma clang loop unroll(full)
                for (uint cc = 0; cc < 16u; ++cc) {
                    float4 a = xv[cc];
                    float4 b = wv[cc];
                    sum += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
                }
            }
        }

        y[((n * H_out + h_out) * W_out + w_out) * OUT_K + t_k] = sum;
    }
}
