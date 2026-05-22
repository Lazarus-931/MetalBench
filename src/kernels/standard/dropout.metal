// dropout (inverted): y = mask * x * (1/(1-p)). float4 grid-stride, 4-way unrolled.
#include <metal_stdlib>
using namespace metal;

kernel void dropout_f32(
    device const float*  X       [[buffer(0)]],
    device const float*  M       [[buffer(1)]],
    device       float*  Y       [[buffer(2)]],
    constant     uint&   N       [[buffer(3)]],
    constant     uint&   grid_size [[buffer(4)]],
    constant     float&  p       [[buffer(5)]],
    uint tid                    [[thread_position_in_grid]])
{
    const float scale = 1.0f / (1.0f - p);
    const device float4* X4 = reinterpret_cast<const device float4*>(X);
    const device float4* M4 = reinterpret_cast<const device float4*>(M);
    device       float4* Y4 = reinterpret_cast<device float4*>(Y);
    const uint n4 = N >> 2;
    const uint gs = grid_size;

    uint i0 = tid;
    uint i1 = i0 + gs;
    uint i2 = i1 + gs;
    uint i3 = i2 + gs;
    if (i3 < n4) {
        float4 x0 = X4[i0]; float4 m0 = M4[i0];
        float4 x1 = X4[i1]; float4 m1 = M4[i1];
        float4 x2 = X4[i2]; float4 m2 = M4[i2];
        float4 x3 = X4[i3]; float4 m3 = M4[i3];
        Y4[i0] = m0 * x0 * scale;
        Y4[i1] = m1 * x1 * scale;
        Y4[i2] = m2 * x2 * scale;
        Y4[i3] = m3 * x3 * scale;
    } else {
        for (uint i = tid; i < n4; i += gs) {
            Y4[i] = M4[i] * X4[i] * scale;
        }
    }
}
