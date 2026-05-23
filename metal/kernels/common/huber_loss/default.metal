#include <metal_stdlib>
using namespace metal;

// Memory-bound elementwise. N=1M, grid=64K threads => 16 elements/thread.
// Use float4 vectorized loads/stores: 4 float4 per thread = 16 elements,
// fully utilizing 128B loads and halving instruction count.
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
    device const float4* pred4 = reinterpret_cast<device const float4*>(pred);
    device const float4* tgt4  = reinterpret_cast<device const float4*>(target);
    device       float4* y4    = reinterpret_cast<device       float4*>(y);
    const float half_delta = 0.5f * delta;

    // Branchless: let m = min(|r|, delta). Then loss = 0.5*m^2 + delta*(|r| - m).
    //   if |r| <= delta: m=|r|, second term=0, result = 0.5*r^2.
    //   if |r| > delta:  m=delta, result = 0.5*delta^2 + delta*(|r| - delta).
    for (uint i = tid; i < N4; i += grid_size) {
        float4 r = pred4[i] - tgt4[i];
        float4 a = fabs(r);
        float4 m = fmin(a, delta);
        y4[i] = fma(0.5f * m, m, delta * (a - m));
    }

    // Tail (N not divisible by 4) — ranks past N4*4
    uint base = N4 << 2;
    for (uint i = base + tid; i < N; i += grid_size) {
        float r = pred[i] - target[i];
        float a = fabs(r);
        float m = fmin(a, delta);
        y[i] = fma(0.5f * m, m, delta * (a - m));
    }
}
