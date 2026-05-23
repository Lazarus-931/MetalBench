// cross_entropy_loss: per-row -sum(y * log_softmax(L))
//                   = lse * sum(y) - sum(y * L)
// where lse = row_max + log(sum_c exp(L_c - row_max))
//
// Fused reductions: pass 1 computes row_max, sum_y, sum_yl simultaneously.
// Pass 2 computes sum_exp.  Then write out[row] = lse*sum_y - sum_yl.
#include <metal_stdlib>
using namespace metal;

kernel void cross_entropy_loss_f32(
    device const float*  L        [[buffer(0)]],
    device const float*  Yh       [[buffer(1)]],
    device       float*  out      [[buffer(2)]],
    constant     uint&   N        [[buffer(3)]],
    constant     uint&   C        [[buffer(4)]],
    uint3 tid3                   [[thread_position_in_threadgroup]],
    uint3 tgid                   [[threadgroup_position_in_grid]])
{
    const uint tid  = tid3.x;
    const uint row  = tgid.y;
    const uint lane = tid & 31u;
    const uint sgid = tid >> 5;

    device const float* lp = L  + row * C;
    device const float* yp = Yh + row * C;

    threadgroup float r0[32]; // max
    threadgroup float r1[32]; // sum_y
    threadgroup float r2[32]; // sum_yl
    threadgroup float r3[32]; // sum_exp
    threadgroup float sc[3];  // [row_max, sum_y, sum_yl] – not strictly needed but keeps clarity

    float lv = lp[tid];
    float yv = yp[tid];

    // ---- pass 1: max, sum(y), sum(y*l) ----
    float mx  = simd_max(lv);
    float sy  = simd_sum(yv);
    float syl = simd_sum(yv * lv);
    if (lane == 0) { r0[sgid] = mx; r1[sgid] = sy; r2[sgid] = syl; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float row_max, sum_y, sum_yl;
    if (sgid == 0) {
        float v0 = r0[lane];
        float v1 = r1[lane];
        float v2 = r2[lane];
        v0 = simd_max(v0);
        v1 = simd_sum(v1);
        v2 = simd_sum(v2);
        if (lane == 0) { sc[0] = v0; sc[1] = v1; sc[2] = v2; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    row_max = sc[0];
    sum_y   = sc[1];
    sum_yl  = sc[2];

    // ---- pass 2: sum(exp(l - row_max)) ----
    float e  = fast::exp(lv - row_max);
    float se = simd_sum(e);
    if (lane == 0) r3[sgid] = se;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = r3[lane];
        v = simd_sum(v);
        if (lane == 0) {
            float lse = row_max + fast::log(v);
            out[row] = lse * sum_y - sum_yl;
        }
    }
}
