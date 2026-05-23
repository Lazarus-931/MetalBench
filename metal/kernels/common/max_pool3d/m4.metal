// max_pool3d M4 variant: NDHWC, K=2, stride=2.
// Vectorized along C with float4 loads/stores; one thread = 4 output elements.
// Tuned for shape (4,32,32,32,32) → (4,16,16,16,32): all dims pow-of-2.
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(1024)]]
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
    const uint C4 = C >> 2;
    const uint total4 = N * D2 * H2 * W2 * C4;

    const uint sW_in4 = C >> 2;
    const uint sH_in4 = (W * C) >> 2;
    const uint sD_in4 = (H * W * C) >> 2;
    const uint sN_in4 = (D * H * W * C) >> 2;

    // Bit-widths for grid-agnostic but power-of-two-friendly index decomposition.
    const uint C4_bits = uint(ctz(C4));
    const uint W2_bits = uint(ctz(W2));
    const uint H2_bits = uint(ctz(H2));
    const uint D2_bits = uint(ctz(D2));

    device const float4* x4 = (device const float4*)x;
    device       float4* y4 = (device       float4*)y;

    for (uint idx = tid; idx < total4; idx += 64u * 1024u) {
        uint q  = idx;
        uint c4 = q & (C4 - 1);     q >>= C4_bits;
        uint w2 = q & (W2 - 1);     q >>= W2_bits;
        uint h2 = q & (H2 - 1);     q >>= H2_bits;
        uint d2 = q & (D2 - 1);     uint n = q >> D2_bits;

        uint base4 = n * sN_in4
                   + (d2 << 1) * sD_in4
                   + (h2 << 1) * sH_in4
                   + (w2 << 1) * sW_in4
                   + c4;

        float4 v000 = x4[base4];
        float4 v001 = x4[base4 + sW_in4];
        float4 v010 = x4[base4 + sH_in4];
        float4 v011 = x4[base4 + sH_in4 + sW_in4];
        float4 v100 = x4[base4 + sD_in4];
        float4 v101 = x4[base4 + sD_in4 + sW_in4];
        float4 v110 = x4[base4 + sD_in4 + sH_in4];
        float4 v111 = x4[base4 + sD_in4 + sH_in4 + sW_in4];

        float4 m = fmax(fmax(fmax(v000, v001), fmax(v010, v011)),
                        fmax(fmax(v100, v101), fmax(v110, v111)));

        y4[idx] = m;
    }
}
