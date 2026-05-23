// avg_pool1d: NLC. mean over window.
#include <metal_stdlib>
using namespace metal;

kernel void avg_pool1d_f32(
    device const float*  x       [[buffer(0)]],
    device       float*  y       [[buffer(1)]],
    constant     uint&   N       [[buffer(2)]],
    constant     uint&   L       [[buffer(3)]],
    constant     uint&   C       [[buffer(4)]],
    constant     uint&   K       [[buffer(5)]],
    constant     uint&   stride  [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint L2 = (L - K) / stride + 1;
    const uint total = N * L2 * C;
    const float inv = 1.0f / float(K);
    for (uint idx = tid; idx < total; idx += 64 * 1024) {
        uint n = idx / (L2 * C); uint r = idx % (L2 * C);
        uint l2 = r / C; uint c = r % C;
        float s = 0.0f;
        for (uint k = 0; k < K; ++k) s += x[(n * L + l2 * stride + k) * C + c];
        y[(n * L2 + l2) * C + c] = s * inv;
    }
}
