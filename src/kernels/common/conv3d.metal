// conv3d: direct convolution, baseline quality.
#include <metal_stdlib>
using namespace metal;

kernel void conv3d_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  w       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   N        [[buffer(3)]],
    constant     uint&   C        [[buffer(4)]],
    constant     uint&   D        [[buffer(5)]],
    constant     uint&   H        [[buffer(6)]],
    constant     uint&   W        [[buffer(7)]],
    constant     uint&   K        [[buffer(8)]],
    constant     uint&   R        [[buffer(9)]],
    constant     uint&   stride   [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    const uint D2 = (D - R) / stride + 1, H2 = (H - R) / stride + 1, W2 = (W - R) / stride + 1;
    const uint total = N * K * D2 * H2 * W2;
    for (uint idx = tid; idx < total; idx += 64*1024) {
        uint n = idx / (K*D2*H2*W2), r = idx % (K*D2*H2*W2);
        uint k = r/(D2*H2*W2), p = r%(D2*H2*W2);
        uint d2 = p/(H2*W2); p %= (H2*W2);
        uint h2 = p/W2, w2 = p%W2;
        float sum = 0;
        for (uint c = 0; c < C; c++)
            for (uint rr = 0; rr < R; rr++)
                for (uint ss = 0; ss < R; ss++)
                    for (uint tt = 0; tt < R; tt++)
                        sum += x[((((n*C+c)*D+d2*stride+rr)*H+h2*stride+ss)*W+w2*stride+tt)] *
                               w[((((k*C+c)*R+rr)*R+ss)*R+tt)];
        y[(((n*K+k)*D2+d2)*H2+h2)*W2+w2] = sum;
    }
}
