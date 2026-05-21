// depthwise_conv2d: NHWC, groups == C. x (N,H,W,C), w (C,R,S,1), y (N,H2,W2,C).
#include <metal_stdlib>
using namespace metal;

kernel void depthwise_conv2d_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  w       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   N       [[buffer(3)]],
    constant     uint&   C       [[buffer(4)]],
    constant     uint&   H       [[buffer(5)]],
    constant     uint&   W       [[buffer(6)]],
    constant     uint&   R       [[buffer(7)]],
    constant     uint&   S       [[buffer(8)]],
    constant     uint&   stride  [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    const uint H2 = (H - R) / stride + 1;
    const uint W2 = (W - S) / stride + 1;
    const uint total = N * H2 * W2 * C;
    for (uint idx = tid; idx < total; idx += 64 * 1024) {
        uint q = idx;
        uint n = q / (H2 * W2 * C);  q %= (H2 * W2 * C);
        uint h2 = q / (W2 * C);      q %= (W2 * C);
        uint w2 = q / C;
        uint c = q % C;
        float sum = 0;
        for (uint rr = 0; rr < R; ++rr)
            for (uint ss = 0; ss < S; ++ss)
                sum += x[((n * H + h2 * stride + rr) * W + w2 * stride + ss) * C + c]
                     * w[(c * R + rr) * S + ss];
        y[((n * H2 + h2) * W2 + w2) * C + c] = sum;
    }
}
