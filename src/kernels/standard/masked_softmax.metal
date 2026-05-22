// masked_softmax: y = softmax(x + mask). Per-row. Assumes C == TG = 1024.
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
    const uint off = row * C;
    const uint lane = t & 31u;
    const uint sg = t >> 5;

    threadgroup float tg_buf[32];

    float val = X[off + t] + M[off + t];

    // max
    float m = simd_max(val);
    if (lane == 0) tg_buf[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rmax = simd_max(tg_buf[lane]);  // every simdgroup computes; we'll broadcast via tg_buf

    // exp + sum (reuse tg_buf after another barrier)
    float ev = fast::exp(val - rmax);
    float s = simd_sum(ev);
    threadgroup_barrier(mem_flags::mem_threadgroup);  // ensure tg_buf reads done
    if (lane == 0) tg_buf[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rsum = simd_sum(tg_buf[lane]);

    Y[off + t] = ev * (1.0f / rsum);
}
