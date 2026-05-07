// depthwise_conv2d: direct convolution, baseline quality.
#include <metal_stdlib>
using namespace metal;

kernel void depthwise_conv2d_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  w       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   N        [[buffer(3)]],
    constant     uint&   C        [[buffer(4)]],
    constant     uint&   H        [[buffer(5)]],
    constant     uint&   W        [[buffer(6)]],
    constant     uint&   R        [[buffer(7)]],
    constant     uint&   S        [[buffer(8)]],
    constant     uint&   stride   [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    const uint H2 = (H - R) / stride + 1, W2 = (W - S) / stride + 1;
    const uint total = N * C * H2 * W2;
    for (uint idx = tid; idx < total; idx += 64*1024) {
        uint n = idx / (C*H2*W2), r = idx % (C*H2*W2);
        uint c = r / (H2*W2), p = r % (H2*W2);
        uint h2 = p/W2, w2 = p%W2;
        float sum = 0;
        for (uint rr = 0; rr < R; rr++)
            for (uint ss = 0; ss < S; ss++)
                sum += x[((n*H+h2*stride+rr)*W+w2*stride+ss)*C+c] * w[(c*R+rr)*S+ss];
        y[((n*C+c)*H2+h2)*W2+w2] = sum;
    }
}
