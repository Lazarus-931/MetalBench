// avg_pool2d: NHWC. K=2, stride=2. float4 along C dim.
#include <metal_stdlib>
using namespace metal;

kernel void avg_pool2d_f32(
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
    const uint C4 = C >> 2;
    const uint total4 = N * H2 * W2 * C4;
    const float inv = 1.0f / float(K * K);

    const uint sW_in = C;
    const uint sH_in = W * C;
    const uint sN_in = H * W * C;

    for (uint idx = tid; idx < total4; idx += 64 * 1024) {
        uint q = idx;
        uint c4 = q % C4;      q /= C4;
        uint w2 = q % W2;      q /= W2;
        uint h2 = q % H2;      uint n = q / H2;

        uint base = n * sN_in + (h2 * stride) * sH_in + (w2 * stride) * sW_in + (c4 << 2);

        float4 v00 = *reinterpret_cast<const device float4*>(x + base);
        float4 v01 = *reinterpret_cast<const device float4*>(x + base + sW_in);
        float4 v10 = *reinterpret_cast<const device float4*>(x + base + sH_in);
        float4 v11 = *reinterpret_cast<const device float4*>(x + base + sH_in + sW_in);

        float4 s = (v00 + v01) + (v10 + v11);
        *reinterpret_cast<device float4*>(y + (idx << 2)) = s * inv;
    }
}
