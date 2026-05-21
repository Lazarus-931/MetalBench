// fused_qkv_projection: y = x @ W. (S, D_in) @ (D_in, 3*D_head).
// M=128, N=192, K=512. Grid 1D, 256 thr/TG.
#include <metal_stdlib>
using namespace metal;

kernel void fused_qkv_projection_f32(
    device const float*  X        [[buffer(0)]],
    device const float*  W        [[buffer(1)]],
    device       float*  Y        [[buffer(2)]],
    constant     uint&   M        [[buffer(3)]],   // 128
    constant     uint&   N        [[buffer(4)]],   // 192
    constant     uint&   K        [[buffer(5)]],   // 512
    uint tid                     [[thread_position_in_grid]])
{
    if (tid >= M * N) return;
    const uint m = tid / N;
    const uint n = tid % N;
    // Vectorize K loop with float4. K=512, multiple of 4.
    float acc = 0.0f;
    device const float* xr = X + m * K;
    // W is (K, N) row-major; column n is strided by N=192. Not float4-friendly.
    // Process K in chunks of 4, but loading W per-k is just 4 separate loads.
    for (uint k = 0; k < K; k += 4) {
        float4 x = *reinterpret_cast<const device float4*>(&xr[k]);
        float w0 = W[(k + 0) * N + n];
        float w1 = W[(k + 1) * N + n];
        float w2 = W[(k + 2) * N + n];
        float w3 = W[(k + 3) * N + n];
        acc += x.x * w0 + x.y * w1 + x.z * w2 + x.w * w3;
    }
    Y[m * N + n] = acc;
}
