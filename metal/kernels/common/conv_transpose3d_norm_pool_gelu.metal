#include <metal_stdlib>
using namespace metal;

static inline float gelu_tanh(float x) {
    const float kBeta  = 0.7978845608028654f;
    const float kAlpha = 0.044715f;
    float x3 = x * x * x;
    float u = kBeta * (x + kAlpha * x3);
    u = clamp(u, -10.0f, 10.0f);
    return 0.5f * x * (1.0f + tanh(u));
}

// One threadgroup per (n,d2,h2,w2). 32 threads = 1 simdgroup, each = 1 channel.
// Loads a 4x4x4x16 input patch into threadgroup memory (with zero padding for
// out-of-bounds), then performs 8 conv-transpose evaluations + LayerNorm +
// AvgPool + GELU.

kernel void conv_transpose3d_norm_pool_gelu_f32(
    device const float* x        [[buffer(0)]],
    device const float* w        [[buffer(1)]],
    device const float* sum_term [[buffer(2)]],
    device       float* y        [[buffer(3)]],
    constant     uint& N         [[buffer(4)]],
    constant     uint& C_in      [[buffer(5)]],
    constant     uint& D         [[buffer(6)]],
    constant     uint& H         [[buffer(7)]],
    constant     uint& W         [[buffer(8)]],
    constant     uint& K         [[buffer(9)]],
    constant     uint& R         [[buffer(10)]],
    constant     float& eps      [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]])
{
    const uint D2 = 5, H2 = 5, W2 = 5;
    const uint Ci = 16, Rc = 3, Dc = 8, Hc = 8, Wc = 8, Kc = 32;

    uint flat = tgid.x;
    uint w2 = flat % W2; flat /= W2;
    uint h2 = flat % H2; flat /= H2;
    uint d2 = flat % D2; uint n = flat / D2;

    uint kk = lid3.x;

    // The input patch we need: d_in = d_full - rd, where d_full in
    // {2*d2, 2*d2+1} and rd in {0,1,2}. So d_in in {2*d2-2, ..., 2*d2+1} -> 4 values.
    // Patch origin in input coords:
    int d_o = int(d2) * 2 - 2;
    int h_o = int(h2) * 2 - 2;
    int w_o = int(w2) * 2 - 2;

    // 4*4*4*16 = 1024 floats; loaded co-operatively by 32 threads (32 each).
    threadgroup float xtile[4 * 4 * 4 * 16];

    // Each thread loads 32 floats.
    for (uint i = 0; i < 32; ++i) {
        uint idx = kk * 32 + i;                  // 0..1023
        uint c  = idx & 15u;
        uint t  = idx >> 4;                      // 0..63 -> (dd,hh,ww) over 4x4x4
        uint ww = t & 3u;
        uint hh = (t >> 2) & 3u;
        uint dd = (t >> 4) & 3u;
        int d_in = d_o + int(dd);
        int h_in = h_o + int(hh);
        int w_in = w_o + int(ww);
        float v = 0.0f;
        if (d_in >= 0 && d_in < int(Dc) &&
            h_in >= 0 && h_in < int(Hc) &&
            w_in >= 0 && w_in < int(Wc)) {
            uint x_base = (((n * Dc + uint(d_in)) * Hc + uint(h_in)) * Wc + uint(w_in)) * Ci + c;
            v = x[x_base];
        }
        xtile[idx] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float bias = sum_term[kk];
    uint wk_base = kk * (Rc * Rc * Rc * Ci);

    float pool_acc = 0.0f;

    // The 8 pool sub-voxels.
    for (uint pd = 0; pd < 2; ++pd) {
        for (uint ph = 0; ph < 2; ++ph) {
            for (uint pw = 0; pw < 2; ++pw) {
                float s = bias;
                // d_full = 2*d2 + pd; d_in = d_full - rd = 2*d2 + pd - rd
                // Within the tile, dd = d_in - d_o = pd - rd + 2. For rd=0..2 -> dd=pd+2,pd+1,pd.
                // So dd ranges {pd, pd+1, pd+2}. Similar for hh,ww.
                for (uint rd = 0; rd < 3; ++rd) {
                    uint dd = pd + 2 - rd;       // always in [0,3]
                    for (uint rh = 0; rh < 3; ++rh) {
                        uint hh = ph + 2 - rh;
                        for (uint rw = 0; rw < 3; ++rw) {
                            uint ww = pw + 2 - rw;
                            uint tile_base = ((dd * 4 + hh) * 4 + ww) * Ci;
                            uint w_base = wk_base + ((rd * Rc + rh) * Rc + rw) * Ci;
                            float4 xv0 = *((threadgroup float4*)(xtile + tile_base + 0));
                            float4 xv1 = *((threadgroup float4*)(xtile + tile_base + 4));
                            float4 xv2 = *((threadgroup float4*)(xtile + tile_base + 8));
                            float4 xv3 = *((threadgroup float4*)(xtile + tile_base + 12));
                            float4 wv0 = *((device const float4*)(w + w_base + 0));
                            float4 wv1 = *((device const float4*)(w + w_base + 4));
                            float4 wv2 = *((device const float4*)(w + w_base + 8));
                            float4 wv3 = *((device const float4*)(w + w_base + 12));
                            s += dot(xv0, wv0) + dot(xv1, wv1) + dot(xv2, wv2) + dot(xv3, wv3);
                        }
                    }
                }
                float sum   = simd_sum(s);
                float sumsq = simd_sum(s * s);
                float mean  = sum * (1.0f / 32.0f);
                float var   = sumsq * (1.0f / 32.0f) - mean * mean;
                float inv   = rsqrt(max(var, 0.0f) + eps);
                pool_acc += (s - mean) * inv;
            }
        }
    }

    float avg = pool_acc * (1.0f / 8.0f);
    float out = gelu_tanh(avg);
    uint y_idx = (((n * D2 + d2) * H2 + h2) * W2 + w2) * Kc + kk;
    y[y_idx] = out;
}
