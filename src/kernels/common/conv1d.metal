// conv1d: direct convolution, baseline quality.
#include <metal_stdlib>
using namespace metal;

kernel void conv1d_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  w       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   N        [[buffer(3)]],
    constant     uint&   C        [[buffer(4)]],
    constant     uint&   L        [[buffer(5)]],
    constant     uint&   K        [[buffer(6)]],
    constant     uint&   R        [[buffer(7)]],
    constant     uint&   stride   [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    const uint L2 = (L - R) / stride + 1;
    const uint total = N * K * L2;
    for (uint idx = tid; idx < total; idx += 64*1024) {
        uint n = idx / (K * L2), r = idx % (K * L2);
        uint k = r / L2, l2 = r % L2;
        float sum = 0;
        for (uint c = 0; c < C; c++)
            for (uint rr = 0; rr < R; rr++)
                sum += x[((n*C+c)*L + l2*stride + rr)] * w[(k*C+c)*R + rr];
        y[(n*K+k)*L2 + l2] = sum;
    }
}
