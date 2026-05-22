// kl_div_loss M2 variant.
// y[i] = t * (log(t) - lp) where t = |target[i]|+1e-6, lp = -|log_pred[i]|.
// Memory-bound pointwise. Grid=64K threads, N=1M. Process via float4 with a
// grid-stride loop; vectorized loads/stores + fast::log keep us close to
// peak bandwidth. atol=1e-3 lets us use fast::log safely.
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(1024)]]
kernel void kl_div_loss_f32(
    device const float*  log_pred  [[buffer(0)]],
    device const float*  target    [[buffer(1)]],
    device       float*  y         [[buffer(2)]],
    constant     uint&   N         [[buffer(3)]],
    constant     uint&   grid_size [[buffer(4)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const uint N4 = N >> 2;
    device const float4* lp4 = reinterpret_cast<device const float4*>(log_pred);
    device const float4* t4  = reinterpret_cast<device const float4*>(target);
    device       float4* y4  = reinterpret_cast<device       float4*>(y);

    const uint stride = grid_size;

    uint i = tid;
    for (; i < N4; i += stride) {
        float4 t  = fabs(t4[i]) + float4(1e-6f);
        float4 lp = -fabs(lp4[i]);
        float4 lg;
        lg.x = fast::log(t.x);
        lg.y = fast::log(t.y);
        lg.z = fast::log(t.z);
        lg.w = fast::log(t.w);
        y4[i] = t * (lg - lp);
    }

    // Tail when N is not a multiple of 4. Hot path is N=1M (divisible).
    if ((N & 3u) != 0u) {
        const uint tail_start = N4 << 2;
        for (uint j = tail_start + tid; j < N; j += stride) {
            float t  = fabs(target[j]) + 1e-6f;
            float lp = -fabs(log_pred[j]);
            y[j] = t * (fast::log(t) - lp);
        }
    }
}
