#include <metal_stdlib>
using namespace metal;

kernel void kl_div_loss_f32(
    device const float*  log_pred  [[buffer(0)]],
    device const float*  target    [[buffer(1)]],
    device       float*  y         [[buffer(2)]],
    constant     uint&   N         [[buffer(3)]],
    constant     uint&   grid_size [[buffer(4)]],
    uint  tid                     [[thread_position_in_grid]])
{
    for (uint i = tid; i < N; i += grid_size) {
        float t = fabs(target[i]) + 1e-6f;
        float lp = -fabs(log_pred[i]);
        y[i] = t * (log(t) - lp);
    }
}
