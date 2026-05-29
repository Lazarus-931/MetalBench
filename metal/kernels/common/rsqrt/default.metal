// rsqrt: y = 1/sqrt(|x|). float4 grid-stride with 4x unroll.
// Batched loads (all loads issued before any store) maximize in-flight memory
// transactions on M2's bandwidth-limited path.
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(1024)]]
kernel void rsqrt_f32(
    device const float*  x         [[buffer(0)]],
    device       float*  y         [[buffer(1)]],
    constant     uint&   N         [[buffer(2)]],
    constant     uint&   grid_size [[buffer(3)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const device float4* x4 = reinterpret_cast<const device float4*>(x);
    device       float4* y4 = reinterpret_cast<device float4*>(y);

    const uint n4  = N >> 2;
    const uint gs  = grid_size;
    const uint gs2 = gs << 1;
    const uint gs3 = gs2 + gs;
    const uint gs4 = gs << 2;

    uint i = tid;
    for (; i + gs3 < n4; i += gs4) {
        float4 v0 = x4[i];
        float4 v1 = x4[i + gs];
        float4 v2 = x4[i + gs2];
        float4 v3 = x4[i + gs3];
        y4[i]       = fast::rsqrt(fabs(v0));
        y4[i + gs]  = fast::rsqrt(fabs(v1));
        y4[i + gs2] = fast::rsqrt(fabs(v2));
        y4[i + gs3] = fast::rsqrt(fabs(v3));
    }
    for (; i < n4; i += gs) {
        float4 v = x4[i];
        y4[i] = fast::rsqrt(fabs(v));
    }
    for (uint k = (n4 << 2) + tid; k < N; k += gs) {
        y[k] = fast::rsqrt(fabs(x[k]));
    }
}
