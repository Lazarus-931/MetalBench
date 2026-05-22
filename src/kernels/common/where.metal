// where: y = cond > 0.5 ? a : b.
// float4 grid-stride with 4x unroll. Batched loads (all loads before any
// store) maximize in-flight memory transactions. select() for exact
// bit-blend (rtol=atol=0).
#include <metal_stdlib>
using namespace metal;

kernel void where_f32(
    device const float*  cond      [[buffer(0)]],
    device const float*  a         [[buffer(1)]],
    device const float*  b         [[buffer(2)]],
    device       float*  y         [[buffer(3)]],
    constant     uint&   N         [[buffer(4)]],
    constant     uint&   grid_size [[buffer(5)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const device float4* cond4 = reinterpret_cast<const device float4*>(cond);
    const device float4* a4    = reinterpret_cast<const device float4*>(a);
    const device float4* b4    = reinterpret_cast<const device float4*>(b);
    device       float4* y4    = reinterpret_cast<device float4*>(y);

    const uint n4 = N >> 2;
    const uint gs = grid_size;
    const uint gs2 = gs << 1;
    const uint gs3 = gs2 + gs;
    const uint gs4 = gs << 2;

    uint i = tid;
    for (; i + gs3 < n4; i += gs4) {
        uint i1 = i + gs;
        uint i2 = i + gs2;
        uint i3 = i + gs3;
        // Batch all 12 float4 loads first to overlap memory latency.
        float4 c0 = cond4[i];  float4 c1 = cond4[i1];
        float4 c2 = cond4[i2]; float4 c3 = cond4[i3];
        float4 a0 = a4[i];     float4 a1 = a4[i1];
        float4 a2 = a4[i2];    float4 a3 = a4[i3];
        float4 b0 = b4[i];     float4 b1 = b4[i1];
        float4 b2 = b4[i2];    float4 b3 = b4[i3];
        y4[i]  = select(b0, a0, c0 > 0.5f);
        y4[i1] = select(b1, a1, c1 > 0.5f);
        y4[i2] = select(b2, a2, c2 > 0.5f);
        y4[i3] = select(b3, a3, c3 > 0.5f);
    }
    for (; i < n4; i += gs) {
        float4 c = cond4[i];
        float4 av = a4[i];
        float4 bv = b4[i];
        y4[i] = select(bv, av, c > 0.5f);
    }
    for (uint k = n4 * 4u + tid; k < N; k += gs) {
        y[k] = cond[k] > 0.5f ? a[k] : b[k];
    }
}
