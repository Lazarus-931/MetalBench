// conv1d: direct NLC convolution. x (N,L,C_in), w (C_out,R,C_in), y (N,L2,C_out).
#include <metal_stdlib>
using namespace metal;

kernel void conv1d_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  w       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   N       [[buffer(3)]],
    constant     uint&   C       [[buffer(4)]],
    constant     uint&   L       [[buffer(5)]],
    constant     uint&   K       [[buffer(6)]],
    constant     uint&   R       [[buffer(7)]],
    constant     uint&   stride  [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    const uint L2 = (L - R) / stride + 1;
    const uint total = N * L2 * K;
    for (uint idx = tid; idx < total; idx += 64 * 1024) {
        uint n = idx / (L2 * K), p = idx % (L2 * K);
        uint l2 = p / K, k = p % K;
        float sum = 0;
        for (uint rr = 0; rr < R; ++rr)
            for (uint c = 0; c < C; ++c)
                sum += x[(n * L + l2 * stride + rr) * C + c] * w[(k * R + rr) * C + c];
        y[(n * L2 + l2) * K + k] = sum;
    }
}
