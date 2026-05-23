// resnet-mini M2: 2 output rows per main-loop iter (16 iters, 16 barriers).
// All 32 simdgroups busy: 32 for conv_b (2 rows × 16 channels), 32 for conv_a lookahead (2 rows × 16 channels),
// remaining threads for stem lookahead. half memory + half4 vector loads.
// h0_ring=12 (non-pow2) to allow stem lookahead 8 rows ahead without aliasing conv_b residual reads.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint H = 32, W = 32, WP = 34, C1 = 16;
constant constexpr uint H0_RING = 12;
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
    //   h0_rows: 12*34*16 halfs = 13056B
    //   ya_rows:  8*34*16 halfs =  8704B
    //   W_a_tg, W_b_tg:           9216B
    //   gap2: 128B
    //   total ≈ 31.1KB
    threadgroup half  h0_rows[H0_RING * WP * C1];
    threadgroup half  ya_rows[YA_RING * WP * C1];
    threadgroup half  W_a_tg[C1 * 9 * C1];
    threadgroup half  W_b_tg[C1 * 9 * C1];
    threadgroup float gap2[2 * C1];

    for (uint i = tid; i < C1 * 9 * C1; i += 1024u) {
        W_a_tg[i] = half(W_a[i]);
        W_b_tg[i] = half(W_b[i]);
    }
    for (uint i = tid; i < H0_RING * WP * C1; i += 1024u) h0_rows[i] = 0.0h;
    for (uint i = tid; i < YA_RING * WP * C1; i += 1024u) ya_rows[i] = 0.0h;
    if (tid < 2u * C1) gap2[tid] = 0.0f;

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
                s += x[xi+0] * W_stem[wi+0]
                   + x[xi+1] * W_stem[wi+1]
                   + x[xi+2] * W_stem[wi+2];
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
    // Invariants at iter h_b start:
    //   ya rows up to h_b+3 valid
    //   h0 rows up to min(H-1, h_b+7) valid
    // This iter performs (all in parallel, single barrier at end):
    //   (A) conv_b on rows h_b, h_b+1 using simd 0..31 (rows split by simd_id>>4)
    //       reads ya[h_b-1..h_b+2], reads h0[h_b, h_b+1] (residual)
    //   (B) conv_a lookahead on rows h_b+4, h_b+5 using simd 0..31
    //       reads h0[h_b+3..h_b+6], writes ya[h_b+4, h_b+5]
    //   (C) stem lookahead on rows h_b+8, h_b+9
    //       reads x, writes h0[h_b+8, h_b+9]
    //
    // Conflict check on h0 (ring=12):
    //   reads: h_b, h_b+1, h_b+3, h_b+4, h_b+5, h_b+6  (slots = those mod 12)
    //   writes: h_b+8, h_b+9                            (slots = (h_b+8)%12, (h_b+9)%12)
    //   For h_b ∈ {0,2,...,30}, slots offset by 8 and 9 vs read slots (max read = h_b+6).
    //   h_b+8 mod 12 vs h_b mod 12 → diff 8 mod 12 ≠ 0. vs h_b+1 → diff 7 ≠ 0. ... all differ. No race.
    //
    // Conflict check on ya (ring=8):
    //   reads (conv_b): h_b-1, h_b, h_b+1, h_b+2
    //   writes (conv_a): h_b+4, h_b+5
    //   slots differ since 4 < 8 and 5 < 8. No race.

    // The kernel function below must do A, B, C concurrently. Because A and B both want simd 0..31,
    // and there are 32 simdgroups total, we INTERLEAVE: each simdgroup performs BOTH a conv_b output
    // and a conv_a output. Conv_b output: row = h_b + (simd_id>>4), c = simd_id&15.
    // Conv_a output: row = h_b+4 + (simd_id>>4), c = simd_id&15. Different row offsets.

    for (uint h_b = 0; h_b < H; h_b += 2u) {
        // ----- A: conv_b -----
        {
            uint row_sel = simd_id >> 4;
            uint c  = simd_id & 15u;
            uint w_ = simd_lane;
            int hh_ = int(h_b + row_sel);
            // Unroll kh: rows are hh_-1, hh_, hh_+1 (always in-bounds for h_b in [0,30] since hh_ in [0,31])
            // hh_-1 only out-of-bounds when h_b==0 && row_sel==0 (hh_=0 → hh_-1=-1).
            // hh_+1 only out-of-bounds when h_b==30 && row_sel==1 (hh_=31 → hh_+1=32).
            half4 acc4 = half4(0.0h);
            int hh0 = hh_ - 1, hh1 = hh_, hh2 = hh_ + 1;
            bool h0ok = hh0 >= 0;
            bool h2ok = hh2 < 32;
            uint slot0 = uint(hh0 & 31) & (YA_RING-1);
            uint slot1 = uint(hh1) & (YA_RING-1);
            uint slot2 = uint(hh2 & 31) & (YA_RING-1);
            // kw=-1,0,1 ; padding via WP (cols 0 and 33 are zero-init).
            threadgroup const half4* wpb = (threadgroup const half4*)&W_b_tg[c * 9u * C1];
            #pragma clang loop unroll(full)
            for (int kw = -1; kw <= 1; ++kw) {
                uint w_off = (w_ + 1u + uint(kw)) * C1;
                if (h0ok) {
                    threadgroup const half4* yap = (threadgroup const half4*)&ya_rows[slot0 * WP * C1 + w_off];
                    threadgroup const half4* wp  = wpb + uint(kw + 1) * 4u;  // kh=-1
                    acc4 += yap[0] * wp[0] + yap[1] * wp[1] + yap[2] * wp[2] + yap[3] * wp[3];
                }
                {
                    threadgroup const half4* yap = (threadgroup const half4*)&ya_rows[slot1 * WP * C1 + w_off];
                    threadgroup const half4* wp  = wpb + (3u + uint(kw + 1)) * 4u;  // kh=0
                    acc4 += yap[0] * wp[0] + yap[1] * wp[1] + yap[2] * wp[2] + yap[3] * wp[3];
                }
                if (h2ok) {
                    threadgroup const half4* yap = (threadgroup const half4*)&ya_rows[slot2 * WP * C1 + w_off];
                    threadgroup const half4* wp  = wpb + (6u + uint(kw + 1)) * 4u;  // kh=+1
                    acc4 += yap[0] * wp[0] + yap[1] * wp[1] + yap[2] * wp[2] + yap[3] * wp[3];
                }
            }
            float s = float(acc4.x) + float(acc4.y) + float(acc4.z) + float(acc4.w);
            s += float(h0_rows[(uint(hh_) % H0_RING) * WP * C1 + (w_ + 1u) * C1 + c]);
            s = fmax(s, 0.0f);
            float row_sum = simd_sum(s);
            if (simd_lane == 0) gap2[row_sel * C1 + c] += row_sum;
        }
        // ----- B: conv_a lookahead -----
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
        }
        // ----- C: stem lookahead -----
        for (uint flat = tid; flat < 2u * W * C1; flat += 1024u) {
            uint h_off = flat / (W * C1);  // 0 or 1
            uint h_ = h_b + 8u + h_off;
            if (h_ >= H) continue;
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
            h0_rows[(h_ % H0_RING) * WP * C1 + (w_ + 1u) * C1 + c] = half(fmax(s, 0.0f));
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
