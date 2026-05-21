// conv3d: direct NDHWC convolution. x (N,D,H,W,C_in), w (C_out,R,R,R,C_in).
#include <metal_stdlib>
using namespace metal;

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
    uint tid [[thread_position_in_grid]])
{
    const uint D2 = (D - R) / stride + 1;
    const uint H2 = (H - R) / stride + 1;
    const uint W2 = (W - R) / stride + 1;
    const uint total = N * D2 * H2 * W2 * K;
    for (uint idx = tid; idx < total; idx += 64 * 1024) {
        uint q = idx;
        uint n = q / (D2 * H2 * W2 * K);   q %= (D2 * H2 * W2 * K);
        uint d2 = q / (H2 * W2 * K);       q %= (H2 * W2 * K);
        uint h2 = q / (W2 * K);            q %= (W2 * K);
        uint w2 = q / K;
        uint k  = q % K;
        float sum = 0;
        for (uint rd = 0; rd < R; ++rd)
            for (uint rh = 0; rh < R; ++rh)
                for (uint rw = 0; rw < R; ++rw)
                    for (uint c = 0; c < C; ++c)
                        sum += x[(((n * D + d2 * stride + rd) * H + h2 * stride + rh) * W + w2 * stride + rw) * C + c]
                             * w[(((k * R + rd) * R + rh) * R + rw) * C + c];
        y[(((n * D2 + d2) * H2 + h2) * W2 + w2) * K + k] = sum;
    }
}
