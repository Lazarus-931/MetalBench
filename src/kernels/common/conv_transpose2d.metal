// conv_transpose2d: NHWC. x (N,H,W,C_in), w (C_out,R,S,C_in), y (N,H_out,W_out,C_out).
// H_out = (H-1)*stride + R; same for W.
#include <metal_stdlib>
using namespace metal;

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
    uint tid [[thread_position_in_grid]])
{
    const uint H_out = (H - 1) * stride + R;
    const uint W_out = (W - 1) * stride + R;
    const uint total = N * H_out * W_out * C_out;
    for (uint idx = tid; idx < total; idx += 64 * 1024) {
        uint q = idx;
        uint n = q / (H_out * W_out * C_out);  q %= (H_out * W_out * C_out);
        uint h_out = q / (W_out * C_out);      q %= (W_out * C_out);
        uint w_out = q / C_out;
        uint k = q % C_out;
        float sum = 0;
        for (uint r = 0; r < R; ++r) {
            int h_in_signed = int(h_out) - int(r);
            if (h_in_signed < 0 || h_in_signed % int(stride) != 0) continue;
            uint h = uint(h_in_signed) / stride;
            if (h >= H) continue;
            for (uint s = 0; s < R; ++s) {
                int w_in_signed = int(w_out) - int(s);
                if (w_in_signed < 0 || w_in_signed % int(stride) != 0) continue;
                uint wi = uint(w_in_signed) / stride;
                if (wi >= W) continue;
                for (uint c = 0; c < C_in; ++c)
                    sum += x[((n * H + h) * W + wi) * C_in + c]
                         * w[((k * R + r) * R + s) * C_in + c];
            }
        }
        y[((n * H_out + h_out) * W_out + w_out) * C_out + k] = sum;
    }
}
