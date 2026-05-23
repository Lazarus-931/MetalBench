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
    const uint HW2C = H2 * W2 * C;
    const uint W2C = W2 * C;
    const uint total = N * HW2C;
    const uint sH = W * C;
    const uint sN = H * sH;

    for (uint idx = tid; idx < total; idx += 64 * 1024) {
        uint q = idx;
        uint n  = q / HW2C;  q -= n * HW2C;
        uint h2 = q / W2C;   q -= h2 * W2C;
        uint w2 = q / C;
        uint c  = q - w2 * C;

        uint wb = c * 9; // R*S = 9
        float w00 = w[wb + 0]; float w01 = w[wb + 1]; float w02 = w[wb + 2];
        float w10 = w[wb + 3]; float w11 = w[wb + 4]; float w12 = w[wb + 5];
        float w20 = w[wb + 6]; float w21 = w[wb + 7]; float w22 = w[wb + 8];

        uint xb = n * sN + (h2 * stride) * sH + (w2 * stride) * C + c;

        float x00 = x[xb           ]; float x01 = x[xb +     C]; float x02 = x[xb + 2*C];
        float x10 = x[xb + sH      ]; float x11 = x[xb + sH + C]; float x12 = x[xb + sH + 2*C];
        float x20 = x[xb + 2*sH    ]; float x21 = x[xb + 2*sH + C]; float x22 = x[xb + 2*sH + 2*C];

        float sum = x00*w00 + x01*w01 + x02*w02
                  + x10*w10 + x11*w11 + x12*w12
                  + x20*w20 + x21*w21 + x22*w22;

        y[idx] = sum;
    }
}
