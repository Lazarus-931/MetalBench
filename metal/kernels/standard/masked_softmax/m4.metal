// masked_softmax M4: y = softmax(x + mask). Per-row. C=TG=1024.
// Optimized: fast::exp, two disjoint tg_buf slots (no middle barrier),
// fused add+softmax, fast::divide.
#include <metal_stdlib>
using namespace metal;

kernel void masked_softmax_f32(
    device const float*  X       [[buffer(0)]],
    device const float*  M       [[buffer(1)]],
    device       float*  Y       [[buffer(2)]],
    constant     uint&   C       [[buffer(3)]],
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint t = tid3.x;
    const uint row = tgid.y;
    const uint off = row * C + t;
    const uint lane = t & 31u;
    const uint sg = t >> 5;

    // Separate slots so we never need to barrier between sum-store and max-read.
    threadgroup float tg_max[32];
    threadgroup float tg_sum[32];

    float val = X[off] + M[off];

    // Per-simdgroup max
    float sm = simd_max(val);
    if (lane == 0) tg_max[sg] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rmax = simd_max(tg_max[lane]);

    float ev = fast::exp(val - rmax);

    float ss = simd_sum(ev);
    if (lane == 0) tg_sum[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rsum = simd_sum(tg_sum[lane]);

    Y[off] = ev * fast::divide(1.0f, rsum);
}
