// resnet-mini for M2: half4-vectorized inner loops over channels.
// Same streaming structure as m4 but with half4 vector loads from TG memory.
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

    // Precompute ya rows 0 and 1 (group=0,1 of 32 simdgroups).
    {
        uint group = simd_id / C1;
        uint c  = simd_id % C1;
        uint w_ = simd_lane;
        int hh_ = int(group);
        half4 acc4 = half4(0.0h);
        for (int kh = -1; kh <= 1; ++kh) {
            int hh = hh_ + kh;
            if (hh < 0 || hh >= 32) continue;
            uint slot_h = uint(hh) % 6u;
            for (int kw = -1; kw <= 1; ++kw) {
                threadgroup const half4* h0p = (threadgroup const half4*)
                    &h0_rows[slot_h * WP * C1 + (w_ + 1u + uint(kw)) * C1];
                threadgroup const half4* wp  = (threadgroup const half4*)
                    &W_a_tg[((c * 3u + uint(kh+1)) * 3u + uint(kw+1)) * C1];
                acc4 += h0p[0] * wp[0] + h0p[1] * wp[1] + h0p[2] * wp[2] + h0p[3] * wp[3];
            }
        }
        float s = float(acc4.x) + float(acc4.y) + float(acc4.z) + float(acc4.w);
        ya_rows[(uint(hh_) & 3u) * WP * C1 + (w_ + 1u) * C1 + c] = half(fmax(s, 0.0f));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main streaming loop. simd 0..15 do conv_b row h_b; simd 16..31 do conv_a lookahead row h_b+2.
    for (uint h_b = 0; h_b < H; ++h_b) {
        if (simd_id < C1) {
            uint c  = simd_id;
            uint w_ = simd_lane;
            int hh_ = int(h_b);
            half4 acc4 = half4(0.0h);
            for (int kh = -1; kh <= 1; ++kh) {
                int hh = hh_ + kh;
                if (hh < 0 || hh >= 32) continue;
                uint slot_h = uint(hh) & 3u;
                for (int kw = -1; kw <= 1; ++kw) {
                    threadgroup const half4* yap = (threadgroup const half4*)
                        &ya_rows[slot_h * WP * C1 + (w_ + 1u + uint(kw)) * C1];
                    threadgroup const half4* wp  = (threadgroup const half4*)
                        &W_b_tg[((c * 3u + uint(kh+1)) * 3u + uint(kw+1)) * C1];
                    acc4 += yap[0] * wp[0] + yap[1] * wp[1] + yap[2] * wp[2] + yap[3] * wp[3];
                }
            }
            float s = float(acc4.x) + float(acc4.y) + float(acc4.z) + float(acc4.w);
            s += float(h0_rows[(h_b % 6u) * WP * C1 + (w_ + 1u) * C1 + c]);
            s = fmax(s, 0.0f);
            float row_sum = simd_sum(s);
            if (simd_lane == 0) gap[c] += row_sum;
        } else {
            uint c  = simd_id - C1;
            uint w_ = simd_lane;
            int hh_ = int(h_b + 2u);
            if (hh_ < int(H)) {
                half4 acc4 = half4(0.0h);
                for (int kh = -1; kh <= 1; ++kh) {
                    int hh = hh_ + kh;
                    if (hh < 0 || hh >= 32) continue;
                    uint slot_h = uint(hh) % 6u;
                    for (int kw = -1; kw <= 1; ++kw) {
                        threadgroup const half4* h0p = (threadgroup const half4*)
                            &h0_rows[slot_h * WP * C1 + (w_ + 1u + uint(kw)) * C1];
                        threadgroup const half4* wp  = (threadgroup const half4*)
                            &W_a_tg[((c * 3u + uint(kh+1)) * 3u + uint(kw+1)) * C1];
                        acc4 += h0p[0] * wp[0] + h0p[1] * wp[1] + h0p[2] * wp[2] + h0p[3] * wp[3];
                    }
                }
                float s = float(acc4.x) + float(acc4.y) + float(acc4.z) + float(acc4.w);
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
