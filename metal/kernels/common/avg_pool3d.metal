// avg_pool3d: NDHWC. K=2, stride=2. M4: float4 along C dim (C=32 is multiple of 4).
#include <metal_stdlib>
using namespace metal;

kernel void avg_pool3d_f32(
    device const float*  x       [[buffer(0)]],
    device       float*  y       [[buffer(1)]],
    constant     uint&   N       [[buffer(2)]],
    constant     uint&   D       [[buffer(3)]],
    constant     uint&   H       [[buffer(4)]],
    constant     uint&   W       [[buffer(5)]],
    constant     uint&   C       [[buffer(6)]],
    constant     uint&   K       [[buffer(7)]],
    constant     uint&   stride  [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    const uint D2 = (D - K) / stride + 1;
    const uint H2 = (H - K) / stride + 1;
    const uint W2 = (W - K) / stride + 1;
    const uint C4 = C >> 2;
    const uint total4 = N * D2 * H2 * W2 * C4;
    const float inv = 1.0f / float(K * K * K);

    const uint sW_in = C;
    const uint sH_in = W * C;
    const uint sD_in = H * W * C;
    const uint sN_in = D * H * W * C;

    for (uint idx = tid; idx < total4; idx += 64 * 1024) {
        uint q = idx;
        uint c4 = q % C4;      q /= C4;
        uint w2 = q % W2;      q /= W2;
        uint h2 = q % H2;      q /= H2;
        uint d2 = q % D2;      uint n = q / D2;

        uint base = n * sN_in + (d2 * stride) * sD_in + (h2 * stride) * sH_in + (w2 * stride) * sW_in + (c4 << 2);

        float4 v000 = *reinterpret_cast<const device float4*>(x + base);
        float4 v001 = *reinterpret_cast<const device float4*>(x + base + sW_in);
        float4 v010 = *reinterpret_cast<const device float4*>(x + base + sH_in);
        float4 v011 = *reinterpret_cast<const device float4*>(x + base + sH_in + sW_in);
        float4 v100 = *reinterpret_cast<const device float4*>(x + base + sD_in);
        float4 v101 = *reinterpret_cast<const device float4*>(x + base + sD_in + sW_in);
        float4 v110 = *reinterpret_cast<const device float4*>(x + base + sD_in + sH_in);
        float4 v111 = *reinterpret_cast<const device float4*>(x + base + sD_in + sH_in + sW_in);

        float4 s = (v000 + v001) + (v010 + v011) + (v100 + v101) + (v110 + v111);
        *reinterpret_cast<device float4*>(y + (idx << 2)) = s * inv;
    }
}
