// log_softmax_cross_entropy: fused log_softmax + NLL. Output (N,).
// Optimization: combine the per-row sum_exp, sum(yv), sum(yv*lv) into a
// single reduction pass after computing row_max. Then
//     loss = lse * sum_yv - sum_yv_lv,
// where lse = row_max + log(sum_exp). This removes one whole reduction
// round trip vs. computing nll = yv*(lse - lv) and reducing it separately.
//
// Disjoint scratch slots per reduction phase to avoid any inter-phase
// races in the cross-simdgroup step.
#include <metal_stdlib>
using namespace metal;

kernel void log_softmax_cross_entropy_f32(
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

    // Disjoint scratch slots for each reduction to make the cross-simdgroup
    // step race-free without needing barriers to "clear" slots.
    threadgroup float red_max[32];
    threadgroup float red_e  [32];
    threadgroup float red_y  [32];
    threadgroup float red_yl [32];
    threadgroup float row_max_s;
    threadgroup float lse_s;

    float lv = lp[tid];
    float yv = yp[tid];

    // ---- Phase A: row_max ----
    float mx = simd_max(lv);
    if (lane == 0) red_max[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = simd_max(red_max[lane]);
        if (lane == 0) row_max_s = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = row_max_s;

    // ---- Phase B: in parallel sum(exp(lv-max)), sum(yv), sum(yv*lv) ----
    float e   = fast::exp(lv - row_max);
    float yvl = yv * lv;

    float s_e  = simd_sum(e);
    float s_y  = simd_sum(yv);
    float s_yl = simd_sum(yvl);
    if (lane == 0) {
        red_e [sgid] = s_e;
        red_y [sgid] = s_y;
        red_yl[sgid] = s_yl;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float ve  = simd_sum(red_e [lane]);
        float vy  = simd_sum(red_y [lane]);
        float vyl = simd_sum(red_yl[lane]);
        if (lane == 0) {
            float lse = row_max + fast::log(ve);
            lse_s = lse;
            out[row] = lse * vy - vyl;
        }
    }
    (void)lse_s; // suppress unused warning; not read by other threads
}
