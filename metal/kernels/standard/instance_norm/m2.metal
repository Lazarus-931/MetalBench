// instance_norm: normalize per-sample per-channel over (H, W).
#include <metal_stdlib>
using namespace metal;

kernel void instance_norm_f32(
    device const float*  X       [[buffer(0)]],
    device       float*  Y       [[buffer(1)]],
    constant     uint&   N       [[buffer(2)]],
    constant     uint&   C       [[buffer(3)]],
    constant     uint&   H       [[buffer(4)]],
    constant     uint&   W       [[buffer(5)]],
    constant     float&  eps     [[buffer(6)]],
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint TG = 1024;
    const uint tid = tid3.x;
    const uint nc = tgid.y;
    const uint HW = H * W;
    device const float* base_in  = X + nc * HW;
    device       float* base_out = Y + nc * HW;

    // Use float4 vectorized loads for the reduction pass
    float sum = 0.0f, sumsq = 0.0f;
    const uint vec_HW = HW / 4;
    for (uint i = tid; i < vec_HW; i += TG) {
        float4 v = ((device const float4*)base_in)[i];
        sum   += v.x + v.y + v.z + v.w;
        sumsq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    // Handle remaining elements (HW not divisible by 4)
    for (uint i = vec_HW * 4 + tid; i < HW; i += TG) {
        float v = base_in[i];
        sum   += v;
        sumsq += v * v;
    }
    sum   = simd_sum(sum);
    sumsq = simd_sum(sumsq);

    threadgroup float tg_s[32], tg_q[32];
    uint sg = tid >> 5;
    if ((tid & 31) == 0) { tg_s[sg] = sum; tg_q[sg] = sumsq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        sum   = tg_s[tid];
        sumsq = tg_q[tid];
        for (uint s = 16; s > 0; s >>= 1) {
            sum   += simd_shuffle_down(sum,   s);
            sumsq += simd_shuffle_down(sumsq, s);
        }
        if (tid == 0) { tg_s[0] = sum; tg_q[0] = sumsq; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = tg_s[0] / float(HW);
    float var  = max(tg_q[0] / float(HW) - mean * mean, 0.0f);
    float inv  = rsqrt(var + eps);
    // Use float4 vectorized stores for the normalization pass
    for (uint i = tid; i < vec_HW; i += TG) {
        float4 v = ((device const float4*)base_in)[i];
        ((device float4*)base_out)[i] = (v - mean) * inv;
    }
    for (uint i = vec_HW * 4 + tid; i < HW; i += TG)
        base_out[i] = (base_in[i] - mean) * inv;
}
