#include <metal_stdlib>
using namespace metal;

kernel void hinge_loss_f32(
    device const float*  pred      [[buffer(0)]],
    device const float*  target    [[buffer(1)]],
    device       float*  y         [[buffer(2)]],
    constant     uint&   N         [[buffer(3)]],
    constant     uint&   grid_size [[buffer(4)]],
    uint  tid                     [[thread_position_in_grid]])
{
    for (uint i = tid; i < N; i += grid_size) {
        y[i] = fmax(1.0f - pred[i] * target[i], 0.0f);
    }
}
