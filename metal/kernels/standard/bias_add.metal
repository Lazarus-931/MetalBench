// bias_add: y[r, c] = x[r, c] + b[c]. float4 grid-stride loop with cached bias.
#include <metal_stdlib>
using namespace metal;

kernel void bias_add_f32(
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
        *reinterpret_cast<device float4*>(&Y[base]) = x + b;
    }
}
