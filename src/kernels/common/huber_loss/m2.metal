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

    for (uint i = tid; i < N4; i += stride) {
        float4 r = p4[i] - t4[i];
        float4 a = fabs(r);
        float4 q = 0.5f * r * r;
        float4 l = d * (a - hd);
        y4[i] = select(l, q, a <= float4(d));
    }

    if ((N & 3u) != 0u) {
        const uint tail = N4 << 2;
        for (uint j = tail + tid; j < N; j += stride) {
            float r = pred[j] - target[j];
            float a = fabs(r);
            float q = 0.5f * r * r;
            float l = d * (a - hd);
            y[j] = a <= d ? q : l;
        }
    }
}
