// matvec: y = A @ x. One threadgroup per row, float4 dot + simd reduce.
#include <metal_stdlib>
using namespace metal;

kernel void matvec_f32(
    device const float*  A  [[buffer(0)]],
    device const float*  x  [[buffer(1)]],
    device       float*  y  [[buffer(2)]],
    constant     uint&   N  [[buffer(3)]],
    uint3 tid              [[thread_position_in_threadgroup]],
    uint3 tgid             [[threadgroup_position_in_grid]])
{
    const uint t = tid.x;
    const uint row = tgid.y;
    
    // Use float4 vectorized loads for A and x
    const uint vec_N = N / 4;
    float4 sum4 = 0.0f;
    
    if (t < vec_N) {
        device const float4* A_vec = (device const float4*)(A + row * N);
        device const float4* x_vec = (device const float4*)x;
        sum4 = A_vec[t] * x_vec[t];
    }
    
    float sum = sum4.x + sum4.y + sum4.z + sum4.w;
    
    // Handle remaining elements (if N not divisible by 4)
    const uint rem_start = vec_N * 4;
    if (t < N - rem_start) {
        sum += A[row * N + rem_start + t] * x[rem_start + t];
    }
    
    float sg_sum = simd_sum(sum);
    
    threadgroup float tg[32];
    const uint sg = t >> 5;
    const uint lane = t & 31;
    if (lane == 0) tg[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (sg == 0) {
        float v = tg[lane];
        v = simd_sum(v);
        if (lane == 0) y[row] = v;
    }
}
