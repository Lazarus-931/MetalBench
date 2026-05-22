// prelu: x if x>0 else alpha[c]*x. Per-channel slope.
#include <metal_stdlib>
using namespace metal;

kernel void prelu_f32(
    device const float*  x         [[buffer(0)]],
    device const float*  alpha     [[buffer(1)]],
    device       float*  y         [[buffer(2)]],
    constant     uint&   N_total   [[buffer(3)]],
    constant     uint&   C         [[buffer(4)]],
    constant     uint&   grid_size [[buffer(5)]],
    uint  tid                     [[thread_position_in_grid]])
{
    for (uint i = tid; i < N_total; i += grid_size) {
        float v = x[i];
        uint c = i % C;
        float a = alpha[c];
        y[i] = v > 0.0f ? v : a * v;
    }
}
