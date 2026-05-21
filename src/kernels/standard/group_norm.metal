// group_norm: normalize per-sample per-group over (CG, H, W) where CG = C/G.
// One threadgroup per (n, g) block, reduces CG*H*W elements.
#include <metal_stdlib>
using namespace metal;

kernel void group_norm_f32(
    device const float*  X       [[buffer(0)]],
    device       float*  Y       [[buffer(1)]],
    constant     uint&   N       [[buffer(2)]],
    constant     uint&   C       [[buffer(3)]],
    constant     uint&   H       [[buffer(4)]],
    constant     uint&   W       [[buffer(5)]],
    constant     uint&   G       [[buffer(6)]],
    constant     float&  eps     [[buffer(7)]],
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint TG = 1024;
    const uint tid = tid3.x;
    const uint ng  = tgid.y;     // 0 .. N*G - 1
    const uint n   = ng / G;
    const uint g   = ng % G;
    const uint CG  = C / G;
    const uint block = CG * H * W;
    device const float* base_in  = X + (n * C + g * CG) * H * W;
    device       float* base_out = Y + (n * C + g * CG) * H * W;

    float sum = 0.0f, sumsq = 0.0f;
    for (uint i = tid; i < block; i += TG) {
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

    float mean = tg_s[0] / float(block);
    float var  = max(tg_q[0] / float(block) - mean * mean, 0.0f);
    float inv  = rsqrt(var + eps);
    for (uint i = tid; i < block; i += TG)
        base_out[i] = (base_in[i] - mean) * inv;
}
