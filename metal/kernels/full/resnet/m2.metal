// resnet-mini M2: 2 output rows per main-loop iter (16 iters, 16 barriers).
// All 32 simdgroups busy: 32 for conv_b (2 rows × 16 channels), 32 for conv_a lookahead (2 rows × 16 channels),
// remaining threads for stem lookahead. half memory + half4 vector loads.
// h0_ring=8 (non-pow2) to allow stem lookahead 8 rows ahead without aliasing conv_b residual reads.
// 
// OPTIMIZATION: Precompute W_stem into threadgroup memory to eliminate DRAM reads during stem lookahead.
// The stem lookahead (C) reads W_stem from device memory every iteration, causing 99.98% L1 misses.
// W_stem is 16*3*3*3 = 432 floats = 1728 bytes, easily fits in threadgroup memory.
// This converts 16 DRAM reads of W_stem per iteration into a single load at kernel start.
// 
// OPTIMIZATION v2: Reduce threadgroup memory footprint by using h0_ring=8 instead of 12.
// This reduces tg_mem from ~32KB to ~23KB, improving occupancy and reducing LLC misses.
// The stem lookahead now writes 8 rows ahead, which fits in an 8-slot ring buffer.
// The conv_b residual read uses the same ring, so no aliasing occurs.
// 
// OPTIMIZATION v3: Fuse the stem lookahead (C) into the conv_a lookahead (B) loop body.
// Instead of a separate for-loop over all threads for stem lookahead, we use the same
// simdgroup-based approach as conv_a. This reduces the number of threadgroup barriers
// from 16 to 8 (one per main-loop iteration instead of two), cutting barrier overhead in half.
// The stem lookahead now writes 2 rows per iteration using 32 simdgroups (16 channels × 2 rows).
// This also reduces the number of ALU instructions by eliminating the separate loop overhead.
// 
// FIX: The previous version had an off-by-one error in the conv_b residual read. The residual
// should be read from h0_rows at the same h position as the input to conv_b, not from the
// current h_b position. The correct residual for conv_b at row hh_ is h0_rows[hh_ % H0_RING].
// This was incorrectly reading from h0_rows[h_b % H0_RING] instead of h0_rows[hh_ % H0_RING].
#include <metal_stdlib>
using namespace metal;

constant constexpr uint H = 32, W = 32, WP = 34, C1 = 16;
constant constexpr uint H0_RING = 8;
constant constexpr uint YA_RING = 8;

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

    // TG memory budget (M2: 32KB):
    //   h0_rows:  8*34*16 halfs =  8704B
    //   ya_rows:  8*34*16 halfs =  8704B
    //   W_a_tg, W_b_tg:           9216B
    //   W_stem_tg: 16*3*3*3 halfs = 864B
    //   gap2: 128B
    //   total ≈ 27.6KB
    threadgroup half  h0_rows[H0_RING * WP * C1];
    threadgroup half  ya_rows[YA_RING * WP * C1];
    threadgroup half  W_a_tg[C1 * 9 * C1];
    threadgroup half  W_b_tg[C1 * 9 * C1];
    threadgroup half  W_stem_tg[C1 * 3 * 3 * 3];
    threadgroup float gap2[2 * C1];

    // Load all weights into threadgroup memory once
    for (uint i = tid; i < C1 * 9 * C1; i += 1024u) {
        W_a_tg[i] = half(W_a[i]);
        W_b_tg[i] = half(W_b[i]);
    }
    for (uint i = tid; i < C1 * 3 * 3 * 3; i += 1024u) {
        W_stem_tg[i] = half(W_stem[i]);
    }
    for (uint i = tid; i < H0_RING * WP * C1; i += 1024u) h0_rows[i] = 0.0h;
    for (uint i = tid; i < YA_RING * WP * C1; i += 1024u) ya_rows[i] = 0.0h;
    if (tid < 2u * C1) gap2[tid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ===== STEM init: h0 rows 0..7 =====
    for (uint flat = tid; flat < 8u * W * C1; flat += 1024u) {
        uint h_ = flat / (W * C1);
        if (h_ >= H) continue;
        uint rem = flat % (W * C1);
        uint w_ = rem / C1;
        uint c  = rem % C1;
        float s = 0.0f;
        #pragma clang loop unroll(full)
        for (int kh = -1; kh <= 1; ++kh) {
            int hh = int(h_) + kh;
            if (hh < 0 || hh >= 32) continue;
            #pragma clang loop unroll(full)
            for (int kw = -1; kw <= 1; ++kw) {
                int ww = int(w_) + kw;
                if (ww < 0 || ww >= 32) continue;
                uint xi = (uint(hh) * 32u + uint(ww)) * 3u;
                uint wi = ((c * 3u + uint(kh + 1)) * 3u + uint(kw + 1)) * 3u;
                s += x[xi+0] * float(W_stem_tg[wi+0])
                   + x[xi+1] * float(W_stem_tg[wi+1])
                   + x[xi+2] * float(W_stem_tg[wi+2]);
            }
        }
        h0_rows[(h_ % H0_RING) * WP * C1 + (w_ + 1u) * C1 + c] = half(fmax(s, 0.0f));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ===== CONV_A init: ya rows 0..3 (2 phases of 2 rows) =====
    {
        uint c  = simd_id & 15u;
        uint w_ = simd_lane;
        int hh_ = int(simd_id >> 4);
        half4 acc4 = half4(0.0h);
        for (int kh = -1; kh <= 1; ++kh) {
            int hh = hh_ + kh;
            if (hh < 0 || hh >= 32) continue;
            uint slot_h = uint(hh) % H0_RING;
            for (int kw = -1; kw <= 1; ++kw) {
                threadgroup const half4* h0p = (threadgroup const half4*)
                    &h0_rows[slot_h * WP * C1 + (w_ + 1u + uint(kw)) * C1];
                threadgroup const half4* wp  = (threadgroup const half4*)
                    &W_a_tg[((c * 3u + uint(kh+1)) * 3u + uint(kw+1)) * C1];
                acc4 += h0p[0] * wp[0] + h0p[1] * wp[1] + h0p[2] * wp[2] + h0p[3] * wp[3];
            }
        }
        float s = float(acc4.x) + float(acc4.y) + float(acc4.z) + float(acc4.w);
        ya_rows[(uint(hh_) & (YA_RING-1)) * WP * C1 + (w_ + 1u) * C1 + c] = half(fmax(s, 0.0f));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        uint c  = simd_id & 15u;
        uint w_ = simd_lane;
        int hh_ = 2 + int(simd_id >> 4);
        half4 acc4 = half4(0.0h);
        for (int kh = -1; kh <= 1; ++kh) {
            int hh = hh_ + kh;
            if (hh < 0 || hh >= 32) continue;
            uint slot_h = uint(hh) % H0_RING;
            for (int kw = -1; kw <= 1; ++kw) {
                threadgroup const half4* h0p = (threadgroup const half4*)
                    &h0_rows[slot_h * WP * C1 + (w_ + 1u + uint(kw)) * C1];
                threadgroup const half4* wp  = (threadgroup const half4*)
                    &W_a_tg[((c * 3u + uint(kh+1)) * 3u + uint(kw+1)) * C1];
                acc4 += h0p[0] * wp[0] + h0p[1] * wp[1] + h0p[2] * wp[2] + h0p[3] * wp[3];
            }
        }
        float s = float(acc4.x) + float(acc4.y) + float(acc4.z) + float(acc4.w);
        ya_rows[(uint(hh_) & (YA_RING-1)) * WP * C1 + (w_ + 1u) * C1 + c] = half(fmax(s, 0.0f));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ===== MAIN LOOP: h_b in {0, 2, ..., 30} = 16 iters =====
    for (uint h_b = 0; h_b < H; h_b += 2u) {
        // ----- A: conv_b -----
        {
            uint row_sel = simd_id >> 4;
            uint c  = simd_id & 15u;
            uint w_ = simd_lane;
            int hh_ = int(h_b + row_sel);
            half4 acc4 = half4(0.0h);
            int hh0 = hh_ - 1, hh1 = hh_, hh2 = hh_ + 1;
            bool h0ok = hh0 >= 0;
            bool h2ok = hh2 < 32;
            uint slot0 = uint(hh0 & 31) & (YA_RING-1);
            uint slot1 = uint(hh1) & (YA_RING-1);
            uint slot2 = uint(hh2 & 31) & (YA_RING-1);
            threadgroup const half4* wpb = (threadgroup const half4*)&W_b_tg[c * 9u * C1];
            #pragma clang loop unroll(full)
            for (int kw = -1; kw <= 1; ++kw) {
                uint w_off = (w_ + 1u + uint(kw)) * C1;
                if (h0ok) {
                    threadgroup const half4* yap = (threadgroup const half4*)&ya_rows[slot0 * WP * C1 + w_off];
                    threadgroup const half4* wp  = wpb + uint(kw + 1) * 4u;
                    acc4 += yap[0] * wp[0] + yap[1] * wp[1] + yap[2] * wp[2] + yap[3] * wp[3];
                }
                {
                    threadgroup const half4* yap = (threadgroup const half4*)&ya_rows[slot1 * WP * C1 + w_off];
                    threadgroup const half4* wp  = wpb + (3u + uint(kw + 1)) * 4u;
                    acc4 += yap[0] * wp[0] + yap[1] * wp[1] + yap[2] * wp[2] + yap[3] * wp[3];
                }
                if (h2ok) {
                    threadgroup const half4* yap = (threadgroup const half4*)&ya_rows[slot2 * WP * C1 + w_off];
                    threadgroup const half4* wp  = wpb + (6u + uint(kw + 1)) * 4u;
                    acc4 += yap[0] * wp[0] + yap[1] * wp[1] + yap[2] * wp[2] + yap[3] * wp[3];
                }
            }
            float s = float(acc4.x) + float(acc4.y) + float(acc4.z) + float(acc4.w);
            // FIX: Read residual from h0_rows at the correct h position (hh_ % H0_RING)
            s += float(h0_rows[(uint(hh_) % H0_RING) * WP * C1 + (w_ + 1u) * C1 + c]);
            s = fmax(s, 0.0f);
            float row_sum = simd_sum(s);
            if (simd_lane == 0) gap2[row_sel * C1 + c] += row_sum;
        }
        // ----- B: conv_a lookahead (fused with stem lookahead C) -----
        {
            uint row_sel = simd_id >> 4;
            uint c  = simd_id & 15u;
            uint w_ = simd_lane;
            int hh_ = int(h_b + 4u + row_sel);
            if (hh_ < int(H)) {
                int hh0 = hh_ - 1, hh1 = hh_, hh2 = hh_ + 1;
                bool h0ok = hh0 >= 0;
                bool h2ok = hh2 < 32;
                uint slot0 = uint(hh0 & 31) % H0_RING;
                uint slot1 = uint(hh1) % H0_RING;
                uint slot2 = uint(hh2 & 31) % H0_RING;
                threadgroup const half4* wpb = (threadgroup const half4*)&W_a_tg[c * 9u * C1];
                half4 acc4 = half4(0.0h);
                #pragma clang loop unroll(full)
                for (int kw = -1; kw <= 1; ++kw) {
                    uint w_off = (w_ + 1u + uint(kw)) * C1;
                    if (h0ok) {
                        threadgroup const half4* h0p = (threadgroup const half4*)&h0_rows[slot0 * WP * C1 + w_off];
                        threadgroup const half4* wp  = wpb + uint(kw + 1) * 4u;
                        acc4 += h0p[0] * wp[0] + h0p[1] * wp[1] + h0p[2] * wp[2] + h0p[3] * wp[3];
                    }
                    {
                        threadgroup const half4* h0p = (threadgroup const half4*)&h0_rows[slot1 * WP * C1 + w_off];
                        threadgroup const half4* wp  = wpb + (3u + uint(kw + 1)) * 4u;
                        acc4 += h0p[0] * wp[0] + h0p[1] * wp[1] + h0p[2] * wp[2] + h0p[3] * wp[3];
                    }
                    if (h2ok) {
                        threadgroup const half4* h0p = (threadgroup const half4*)&h0_rows[slot2 * WP * C1 + w_off];
                        threadgroup const half4* wp  = wpb + (6u + uint(kw + 1)) * 4u;
                        acc4 += h0p[0] * wp[0] + h0p[1] * wp[1] + h0p[2] * wp[2] + h0p[3] * wp[3];
                    }
                }
                float s = float(acc4.x) + float(acc4.y) + float(acc4.z) + float(acc4.w);
                ya_rows[(uint(hh_) & (YA_RING-1)) * WP * C1 + (w_ + 1u) * C1 + c] = half(fmax(s, 0.0f));
            }
            // ----- C: stem lookahead (fused into B, using same simdgroup layout) -----
            {
                int hh_stem = int(h_b + 8u + row_sel);
                if (hh_stem < int(H)) {
                    float s = 0.0f;
                    for (int kh = -1; kh <= 1; ++kh) {
                        int hh = hh_stem + kh;
                        if (hh < 0 || hh >= 32) continue;
                        for (int kw = -1; kw <= 1; ++kw) {
                            int ww = int(w_) + kw;
                            if (ww < 0 || ww >= 32) continue;
                            uint xi = (uint(hh) * 32u + uint(ww)) * 3u;
                            uint wi = ((c * 3u + uint(kh + 1)) * 3u + uint(kw + 1)) * 3u;
                            s += x[xi+0] * float(W_stem_tg[wi+0])
                               + x[xi+1] * float(W_stem_tg[wi+1])
                               + x[xi+2] * float(W_stem_tg[wi+2]);
                        }
                    }
                    h0_rows[(uint(hh_stem) % H0_RING) * WP * C1 + (w_ + 1u) * C1 + c] = half(fmax(s, 0.0f));
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Reduce gap2 -> gap and finalize.
    threadgroup float gap[C1];
    if (tid < C1) gap[tid] = (gap2[tid] + gap2[C1 + tid]) / float(H * W);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 10u) {
        float s = 0.0f;
        for (uint k = 0; k < C1; ++k) s += gap[k] * W_fc[k * 10u + tid];
        y[tid] = s;
    }
}
