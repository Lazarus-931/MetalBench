// bias_gelu: y = gelu(x + b). float4 with cached bias in threadgroup memory.
// Uses tanh-approximation of GELU for speed.
#include <metal_stdlib>
using namespace metal;

constant constexpr float gelu_k = 0.79788456f; // sqrt(2/pi)

static inline float4 gelu_tanh4(float4 v) {
    float4 t = gelu_k * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + tanh(t));
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
        *reinterpret_cast<device float4*>(&Y[base]) = gelu_tanh4(z);
    }
}
