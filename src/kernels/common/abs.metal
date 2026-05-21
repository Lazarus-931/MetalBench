#include <metal_stdlib>
using namespace metal;

kernel void abs_f32(
    device const float*  x         [[buffer(0)]],
    device       float*  y         [[buffer(1)]],
    constant     uint&   N         [[buffer(2)]],
    constant     uint&   grid_size [[buffer(3)]],
    uint  tid                     [[thread_position_in_grid]])
{
    for (uint i = tid; i < N; i += grid_size) {
        y[i] = fabs(x[i]);
    }
}
