// conv_transpose2d: direct convolution, baseline quality.
#include <metal_stdlib>
using namespace metal;

kernel void conv_transpose2d_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  w       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   N        [[buffer(3)]],
    constant     uint&   C        [[buffer(4)]],
    constant     uint&   H        [[buffer(5)]],
    constant     uint&   W        [[buffer(6)]],
    constant     uint&   K        [[buffer(7)]],
    constant     uint&   R        [[buffer(8)]],
    constant     uint&   stride   [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    const uint H2 = (H - 1) * stride + R, W2 = (W - 1) * stride + R;
    const uint total = N * K * H2 * W2;
    for (uint idx = tid; idx < total; idx += 64*1024) {
        uint n = idx/(K*H2*W2), r = idx%(K*H2*W2);
        uint k = r/(H2*W2), p = r%(H2*W2);
        uint h2 = p/W2, w2 = p%W2;
        float sum = 0;
        for (uint c = 0; c < C; c++)
            for (uint rr = 0; rr < R; rr++)
                for (uint ss = 0; ss < R; ss++) {
                    int hi = (int)h2 - (int)rr, wi = (int)w2 - (int)ss;
                    if (hi >= 0 && wi >= 0 && hi % stride == 0 && wi % stride == 0) {
                        hi /= stride; wi /= stride;
                        if (hi < (int)H && wi < (int)W)
                            sum += x[((n*H+hi)*W+wi)*C+c] * w[((c*K+k)*R+rr)*R+ss];
                    }
                }
        y[((n*K+k)*H2+h2)*W2+w2] = sum;
    }
}
