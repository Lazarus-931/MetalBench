// add_norm: y = layer_norm(x + residual). D=1024, TG=1024.
// Single-simdgroup-per-row: only sg 0 (32 lanes) processes the row.
// Each active lane handles 8 float4 = 32 elements -> 1024 floats per row.
// Cache add-result in 4KB threadgroup memory instead of per-thread registers.
// IMPORTANT: registry TG=1024 must match PSO max_thr/tg. Storing the cache in
// registers (float4[8] per thread) drops max_thr/tg below 1024 on M2 because
// the compiler reserves register space for all 1024 launched threads even
// though only 32 do work — the kernel then silently fails to dispatch
// (0.000ms timing, large max_err). Threadgroup memory sidesteps this.
#include <metal_stdlib>
using namespace metal;

kernel void add_norm_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  res     [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant     uint&   D       [[buffer(3)]],
    constant     float&  eps     [[buffer(4)]],
    uint3 tid                   [[thread_position_in_threadgroup]],
    uint3 tgid                  [[threadgroup_position_in_grid]])
{
    const uint t = tid.x;
    threadgroup float4 cache[256];   // 4KB: full row of (x+res) in shared mem

    if (t < 32u) {
        const uint row = tgid.y;
        device const float4* xr = (device const float4*)(x + row * D);
        device const float4* rr = (device const float4*)(res + row * D);

        float ssum = 0.0f, ssq = 0.0f;
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            uint i = k * 32u + t;            // strided coalesced
            float4 u = xr[i] + rr[i];
            cache[i] = u;
            ssum += u.x + u.y + u.z + u.w;
            ssq  += dot(u, u);
        }
        ssum = simd_sum(ssum);
        ssq  = simd_sum(ssq);

        const float invD    = 1.0f / float(D);
        const float mean    = ssum * invD;
        const float var     = max(ssq * invD - mean * mean, 0.0f);
        const float inv_std = rsqrt(var + eps);

        device float4* yr = (device float4*)(y + row * D);
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            uint i = k * 32u + t;
            yr[i] = (cache[i] - mean) * inv_std;
        }
    }
}
