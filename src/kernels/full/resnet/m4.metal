// resnet-mini for M4 v10: full h0 precomputed in TG fp16 (32KB-ε). Stream ya 3 rows.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint H = 32, W = 32, WP = 34, C1 = 16;

kernel void resnet_f32(
    device const float* x       [[buffer(0)]],
    device const float* W_stem  [[buffer(1)]],
    device const float* W_a     [[buffer(2)]],
    device const float* W_b     [[buffer(3)]],
    device const float* W_fc    [[buffer(4)]],
    device       float* y       [[buffer(5)]],
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint  simd_lane             [[thread_index_in_simdgroup]],
    uint  simd_id               [[simdgroup_index_in_threadgroup]])
{
    const uint tid = tid3.x;

    threadgroup half  h0_rows[6 * WP * C1];
    threadgroup half  ya_rows[4 * WP * C1];
    threadgroup half  W_a_tg[C1 * 9 * C1];
    threadgroup half  W_b_tg[C1 * 9 * C1];
    threadgroup float gap[C1];

    for (uint i = tid; i < C1 * 9 * C1; i += 1024u) {
        W_a_tg[i] = half(W_a[i]);
        W_b_tg[i] = half(W_b[i]);
    }
    for (uint i = tid; i < 6u * WP * C1; i += 1024u) h0_rows[i] = 0.0h;
    for (uint i = tid; i < 4u * WP * C1; i += 1024u) ya_rows[i] = 0.0h;
    if (tid < C1) gap[tid] = 0.0f;

    #define COMPUTE_H0_ROW(h_val) \
    do { \
        int hh_ = (h_val); \
        if (tid < W * C1) { \
            uint w_ = tid / C1; \
            uint c  = tid % C1; \
            float s = 0.0f; \
            for (int kh = -1; kh <= 1; ++kh) { \
                int hh = hh_ + kh; \
                if (hh < 0 || hh >= 32) continue; \
                for (int kw = -1; kw <= 1; ++kw) { \
                    int ww = int(w_) + kw; \
                    if (ww < 0 || ww >= 32) continue; \
                    uint xi = (uint(hh) * 32u + uint(ww)) * 3u; \
                    uint wi = ((c * 3u + uint(kh + 1)) * 3u + uint(kw + 1)) * 3u; \
                    s += x[xi+0] * W_stem[wi+0] \
                       + x[xi+1] * W_stem[wi+1] \
                       + x[xi+2] * W_stem[wi+2]; \
                } \
            } \
            h0_rows[(uint(hh_) % 6u) * WP * C1 + (w_ + 1u) * C1 + c] = half(fmax(s, 0.0f)); \
        } \
    } while(0)

    for (uint flat = tid; flat < 6u * W * C1; flat += 1024u) {
        uint h_ = flat / (W * C1);
        uint rem = flat % (W * C1);
        uint w_ = rem / C1;
        uint c  = rem % C1;
        float s = 0.0f;
        for (int kh = -1; kh <= 1; ++kh) {
            int hh = int(h_) + kh;
            if (hh < 0 || hh >= 32) continue;
            for (int kw = -1; kw <= 1; ++kw) {
                int ww = int(w_) + kw;
                if (ww < 0 || ww >= 32) continue;
                uint xi = (uint(hh) * 32u + uint(ww)) * 3u;
                uint wi = ((c * 3u + uint(kh + 1)) * 3u + uint(kw + 1)) * 3u;
                s += x[xi+0] * W_stem[wi+0]
                   + x[xi+1] * W_stem[wi+1]
                   + x[xi+2] * W_stem[wi+2];
            }
        }
        h0_rows[(h_ % 6u) * WP * C1 + (w_ + 1u) * C1 + c] = half(fmax(s, 0.0f));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        uint group = simd_id / C1;
        uint c  = simd_id % C1;
        uint w_ = simd_lane;
        int hh_ = int(group);
        float s = 0.0f;
        for (int kh = -1; kh <= 1; ++kh) {
            int hh = hh_ + kh;
            if (hh < 0 || hh >= 32) continue;
            uint slot_h = uint(hh) % 6u;
            for (int kw = -1; kw <= 1; ++kw) {
                threadgroup const half* h0p = &h0_rows[slot_h * WP * C1 + (w_ + 1u + uint(kw)) * C1];
                threadgroup const half* wp  = &W_a_tg[((c * 3u + uint(kh+1)) * 3u + uint(kw+1)) * C1];
                threadgroup const half2* h0p2 = (threadgroup const half2*)h0p;
                threadgroup const half2* wp2  = (threadgroup const half2*)wp;
                for (uint ci = 0; ci < C1/2; ++ci) {
                    float2 a = float2(h0p2[ci]);
                    float2 b = float2(wp2[ci]);
                    s = fma(a.x, b.x, s);
                    s = fma(a.y, b.y, s);
                }
            }
        }
        ya_rows[(uint(hh_) & 3u) * WP * C1 + (w_ + 1u) * C1 + c] = half(fmax(s, 0.0f));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint h_b = 0; h_b < H; ++h_b) {
        if (simd_id < C1) {
            uint c  = simd_id;
            uint w_ = simd_lane;
            int hh_ = int(h_b);
            float s = 0.0f;
            for (int kh = -1; kh <= 1; ++kh) {
                int hh = hh_ + kh;
                if (hh < 0 || hh >= 32) continue;
                uint slot_h = uint(hh) & 3u;
                for (int kw = -1; kw <= 1; ++kw) {
                    threadgroup const half* yap = &ya_rows[slot_h * WP * C1 + (w_ + 1u + uint(kw)) * C1];
                    threadgroup const half* wp  = &W_b_tg[((c * 3u + uint(kh+1)) * 3u + uint(kw+1)) * C1];
                    threadgroup const half2* yap2 = (threadgroup const half2*)yap;
                    threadgroup const half2* wp2  = (threadgroup const half2*)wp;
                    for (uint ci = 0; ci < C1/2; ++ci) {
                        float2 a = float2(yap2[ci]);
                        float2 b = float2(wp2[ci]);
                        s = fma(a.x, b.x, s);
                        s = fma(a.y, b.y, s);
                    }
                }
            }
            s += float(h0_rows[(h_b % 6u) * WP * C1 + (w_ + 1u) * C1 + c]);
            s = fmax(s, 0.0f);
            float row_sum = simd_sum(s);
            if (simd_lane == 0) gap[c] += row_sum;
        } else {
            uint c  = simd_id - C1;
            uint w_ = simd_lane;
            int hh_ = int(h_b + 2u);
            if (hh_ < int(H)) {
                float s = 0.0f;
                for (int kh = -1; kh <= 1; ++kh) {
                    int hh = hh_ + kh;
                    if (hh < 0 || hh >= 32) continue;
                    uint slot_h = uint(hh) % 6u;
                    for (int kw = -1; kw <= 1; ++kw) {
                        threadgroup const half* h0p = &h0_rows[slot_h * WP * C1 + (w_ + 1u + uint(kw)) * C1];
                        threadgroup const half* wp  = &W_a_tg[((c * 3u + uint(kh+1)) * 3u + uint(kw+1)) * C1];
                        threadgroup const half2* h0p2 = (threadgroup const half2*)h0p;
                        threadgroup const half2* wp2  = (threadgroup const half2*)wp;
                        for (uint ci = 0; ci < C1/2; ++ci) {
                            float2 a = float2(h0p2[ci]);
                            float2 b = float2(wp2[ci]);
                            s = fma(a.x, b.x, s);
                            s = fma(a.y, b.y, s);
                        }
                    }
                }
                ya_rows[(uint(hh_) & 3u) * WP * C1 + (w_ + 1u) * C1 + c] = half(fmax(s, 0.0f));
            }
        }
        if (h_b + 5 < H) {
            COMPUTE_H0_ROW(int(h_b + 5));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < C1) gap[tid] = gap[tid] / float(H * W);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 10u) {
        float s = 0.0f;
        for (uint k = 0; k < C1; ++k) s += gap[k] * W_fc[k * 10u + tid];
        y[tid] = s;
    }
}
