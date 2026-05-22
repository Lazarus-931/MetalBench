// max_pool3d: NDHWC. K=2, stride=2 specialization.
#include <metal_stdlib>
using namespace metal;

kernel void max_pool3d_f32(
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
    const uint total = N * D2 * H2 * W2 * C;

    // Strides (in elements) into input tensor x of shape (N, D, H, W, C)
    const uint sW_in = C;
    const uint sH_in = W * C;
    const uint sD_in = H * W * C;
    const uint sN_in = D * H * W * C;

    for (uint idx = tid; idx < total; idx += 64 * 1024) {
        uint q = idx;
        uint c  = q % C;       q /= C;
        uint w2 = q % W2;      q /= W2;
        uint h2 = q % H2;      q /= H2;
        uint d2 = q % D2;      uint n = q / D2;

        uint base = n * sN_in + (d2 * stride) * sD_in + (h2 * stride) * sH_in + (w2 * stride) * sW_in + c;

        // K=2 unrolled
        float v000 = x[base];
        float v001 = x[base + sW_in];
        float v010 = x[base + sH_in];
        float v011 = x[base + sH_in + sW_in];
        float v100 = x[base + sD_in];
        float v101 = x[base + sD_in + sW_in];
        float v110 = x[base + sD_in + sH_in];
        float v111 = x[base + sD_in + sH_in + sW_in];

        float m = fmax(fmax(fmax(v000, v001), fmax(v010, v011)),
                       fmax(fmax(v100, v101), fmax(v110, v111)));

        y[idx] = m;
    }
}
