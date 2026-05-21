// bias_gelu: y = gelu(x + b). float4 with cached bias in threadgroup memory.
#include <metal_stdlib>
using namespace metal;


// erf-based GELU using polynomial approx (A&S 7.1.26).
static inline float gelu_erf_approx(float x) {
    const float k = 0.70710678f; // 1/sqrt(2)
    float z = x * k;
    float t = 1.0f / (1.0f + 0.3275911f * fabs(z));
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t)
              + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f)
              * t * exp(-z * z);
    float erfz = copysign(y, z);
    return 0.5f * x * (1.0f + erfz);
}

kernel void bias_gelu_f32(
    device const float*  X        [[buffer(0)]],
    device const float*  B        [[buffer(1)]],
    device       float*  Y        [[buffer(2)]],
    constant     uint&   N_total  [[buffer(3)]],
    constant     uint&   C        [[buffer(4)]],
    constant     uint&   grid_size [[buffer(5)]],
    uint tid                       [[thread_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]])
{
    threadgroup float bias_tg[1024];
    for (uint i = lid; i < C; i += tg_size) bias_tg[i] = B[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n4 = N_total >> 2;
    const uint Cmask = C - 1;
    for (uint i = tid; i < n4; i += grid_size) {
        uint base = i << 2;
        float4 x = *reinterpret_cast<const device float4*>(&X[base]);
        float4 b = *reinterpret_cast<const threadgroup float4*>(&bias_tg[base & Cmask]);
        float4 z = x + b;
        float4 r;
        r.x = gelu_erf_approx(z.x);
        r.y = gelu_erf_approx(z.y);
        r.z = gelu_erf_approx(z.z);
        r.w = gelu_erf_approx(z.w);
        *reinterpret_cast<device float4*>(&Y[base]) = r;
    }
}
