// dropout (inverted): y = mask * x * (1/(1-p)).
// N=1M, n4=256K, grid=64K -> exactly 4 float4 per thread (16 floats = 64B).
// Block layout (contiguous): each simdgroup issues a 2 KB coalesced device
// read; 4 such requests fly per thread to saturate memory parallelism while
// keeping per-thread cache locality at one cache line per stream.
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

    const uint base = tid * 4u;
    if (base + 3u < n4) {
        float4 x0 = X4[base + 0];
        float4 x1 = X4[base + 1];
        float4 x2 = X4[base + 2];
        float4 x3 = X4[base + 3];
        float4 m0 = M4[base + 0];
        float4 m1 = M4[base + 1];
        float4 m2 = M4[base + 2];
        float4 m3 = M4[base + 3];
        Y4[base + 0] = m0 * x0 * scale;
        Y4[base + 1] = m1 * x1 * scale;
        Y4[base + 2] = m2 * x2 * scale;
        Y4[base + 3] = m3 * x3 * scale;
        return;
    }
    for (uint i = tid; i < n4; i += gs) {
        Y4[i] = M4[i] * X4[i] * scale;
    }
}
