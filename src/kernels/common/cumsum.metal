// cumsum: Hillis-Steele parallel prefix scan per row.
#include <metal_stdlib>
using namespace metal;

kernel void cumsum_f32(
    device const float*  x  [[buffer(0)]],
    device       float*  y  [[buffer(1)]],
    constant     uint&   N  [[buffer(2)]],
    uint3 tid              [[thread_position_in_threadgroup]],
    uint3 tgid             [[threadgroup_position_in_grid]])
{
    threadgroup float tmp[1024];
    const uint t = tid.x;
    const uint row = tgid.y;
    tmp[t] = x[row * N + t];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1; stride < N; stride <<= 1) {
        float val = tmp[t];
        if (t >= stride) val += tmp[t - stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tmp[t] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    y[row * N + t] = tmp[t];
}
