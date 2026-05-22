#include <metal_stdlib>
using namespace metal;

kernel void huber_loss_f32(
    device const float*  pred      [[buffer(0)]],
    device const float*  target    [[buffer(1)]],
    device       float*  y         [[buffer(2)]],
    constant     uint&   N         [[buffer(3)]],
    constant     uint&   grid_size [[buffer(4)]],
    constant     float&  delta     [[buffer(5)]],
    uint  tid                     [[thread_position_in_grid]])
{
    for (uint i = tid; i < N; i += grid_size) {
        float r = pred[i] - target[i];
        float a = fabs(r);
        float q = 0.5f * r * r;
        float l = delta * (a - 0.5f * delta);
        y[i] = a <= delta ? q : l;
    }
}
