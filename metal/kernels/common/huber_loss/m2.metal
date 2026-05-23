// huber_loss M2 variant.
// y[i] = r^2/2 if |r|<=delta else delta*(|r|-delta/2), r = pred[i]-target[i].
// Memory-bound pointwise (N=1M f32, 3 buffers => 12MB). float4 vectorized
// grid-stride loop, no special functions needed (atol=1e-6, exact ops).
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(1024)]]
kernel void huber_loss_f32(
    device const float*  pred      [[buffer(0)]],
    device const float*  target    [[buffer(1)]],
    device       float*  y         [[buffer(2)]],
    constant     uint&   N         [[buffer(3)]],
    constant     uint&   grid_size [[buffer(4)]],
    constant     float&  delta     [[buffer(5)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const uint N4 = N >> 2;
    device const float4* p4 = reinterpret_cast<device const float4*>(pred);
    device const float4* t4 = reinterpret_cast<device const float4*>(target);
    device       float4* y4 = reinterpret_cast<device       float4*>(y);

    const uint stride = grid_size;
    const float d  = delta;
    const float hd = 0.5f * delta;

    // Branchless identity: huber(r) = a*|r| - a^2/2, where a = min(|r|, delta).
    //   |r|<=delta: a=|r|  →  |r|^2 - |r|^2/2 = r^2/2   = q
    //   |r|> delta: a=delta → delta*|r| - delta^2/2     = l
    const float4 dv = float4(d);
    const uint N8 = N4 & ~1u;
    for (uint i = tid * 2u; i < N8; i += stride * 2u) {
        float4 r0 = p4[i]      - t4[i];
        float4 r1 = p4[i + 1u] - t4[i + 1u];
        float4 ar0 = fabs(r0), ar1 = fabs(r1);
        float4 a0 = fmin(ar0, dv), a1 = fmin(ar1, dv);
        y4[i]      = a0 * (ar0 - 0.5f * a0);
        y4[i + 1u] = a1 * (ar1 - 0.5f * a1);
    }
    if ((N4 & 1u) != 0u && tid == 0u) {
        const uint i = N4 - 1u;
        float4 r = p4[i] - t4[i];
        float4 ar = fabs(r);
        float4 a = fmin(ar, dv);
        y4[i] = a * (ar - 0.5f * a);
    }

    if ((N & 3u) != 0u) {
        const uint tail = N4 << 2;
        for (uint j = tail + tid; j < N; j += stride) {
            float r = pred[j] - target[j];
            float ar = fabs(r);
            float a = fmin(ar, d);
            y[j] = a * (ar - 0.5f * a);
        }
    }
}
