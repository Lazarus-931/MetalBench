// max_pool2d: NHWC. K=2, stride=2 specialization. float4 vectorized across C.
#include <metal_stdlib>
using namespace metal;

kernel void max_pool2d_f32(
    device const float*  x       [[buffer(0)]],
    device       float*  y       [[buffer(1)]],
    constant     uint&   N       [[buffer(2)]],
    constant     uint&   H       [[buffer(3)]],
    constant     uint&   W       [[buffer(4)]],
    constant     uint&   C       [[buffer(5)]],
    constant     uint&   K       [[buffer(6)]],
    constant     uint&   stride  [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    const uint H2 = (H - K) / stride + 1;
    const uint W2 = (W - K) / stride + 1;
    const uint C4 = C / 4;
    const uint total4 = N * H2 * W2 * C4;

    const uint sW_in4 = C4;
    const uint sH_in4 = W * C4;
    const uint sN_in4 = H * W * C4;

    device const float4* x4 = (device const float4*)x;
    device       float4* y4 = (device       float4*)y;

    for (uint idx = tid; idx < total4; idx += 64 * 1024) {
        uint q = idx;
        uint c4 = q % C4;      q /= C4;
        uint w2 = q % W2;      q /= W2;
        uint h2 = q % H2;      uint n = q / H2;

        uint base = n * sN_in4 + (h2 * stride) * sH_in4 + (w2 * stride) * sW_in4 + c4;

        float4 v00 = x4[base];
        float4 v01 = x4[base + sW_in4];
        float4 v10 = x4[base + sH_in4];
        float4 v11 = x4[base + sH_in4 + sW_in4];

        y4[idx] = fmax(fmax(v00, v01), fmax(v10, v11));
    }
}
