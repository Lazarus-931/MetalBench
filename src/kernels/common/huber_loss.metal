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
    // Unroll 4× float4 per iteration (16 elements/thread for N=1M, grid=64K).
    uint i = tid;
    const uint step = grid_size;
    const uint step2 = step << 1;
    const uint step3 = step + step2;
    const uint step4 = step << 2;
    while (i + step3 < N4) {
        float4 p0 = pred4[i];
        float4 p1 = pred4[i + step];
        float4 p2 = pred4[i + step2];
        float4 p3 = pred4[i + step3];
        float4 t0 = tgt4[i];
        float4 t1 = tgt4[i + step];
        float4 t2 = tgt4[i + step2];
        float4 t3 = tgt4[i + step3];
        float4 r0 = p0 - t0, r1 = p1 - t1, r2 = p2 - t2, r3 = p3 - t3;
        float4 a0 = fabs(r0), a1 = fabs(r1), a2 = fabs(r2), a3 = fabs(r3);
        float4 m0 = fmin(a0, delta), m1 = fmin(a1, delta), m2 = fmin(a2, delta), m3 = fmin(a3, delta);
        y4[i]         = fma(0.5f * m0, m0, delta * (a0 - m0));
        y4[i + step]  = fma(0.5f * m1, m1, delta * (a1 - m1));
        y4[i + step2] = fma(0.5f * m2, m2, delta * (a2 - m2));
        y4[i + step3] = fma(0.5f * m3, m3, delta * (a3 - m3));
        i += step4;
    }
    for (; i < N4; i += step) {
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
