// resnet-mini for M4 v11: conv_b refactored to simdgroup_matrix<float,8,8> MMA.
// conv_a + stem unchanged (unroll + half acc on C1=16). conv_b: 4 simdgroups tile
// 8 spatial × 16 chan output via 9 taps × (8x8 K-tiles) of MMA. Inputs/weights
// staged through TG as float (8 spatial wide, padded) so we can load directly
// into simdgroup_matrix<float,8,8> with simdgroup_load.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
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
    // W_b restaged as float, layout [kh,kw,c_in_outer(2 of 8),c_out_outer(2 of 8), 8c_in × 8c_out row-major]
    // Simpler: keep W_b in row-major [9 * C1 * C1] float and load 8x8 tiles directly.
    threadgroup float W_b_tg[9 * C1 * C1];
    threadgroup float gap[C1];

    // ya halo as float for MMA load (3 rows × 34 width × 16 ch), small ring of 4 rows
    threadgroup float ya_f[4 * WP * C1];

    for (uint i = tid; i < C1 * 9 * C1; i += 1024u) {
        W_a_tg[i] = half(W_a[i]);
        W_b_tg[i] = W_b[i];
    }
    for (uint i = tid; i < 6u * WP * C1; i += 1024u) h0_rows[i] = 0.0h;
    for (uint i = tid; i < 4u * WP * C1; i += 1024u) { ya_rows[i] = 0.0h; ya_f[i] = 0.0f; }
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

    // conv_a: compute ya rows 0..2 (need 3 rows of ya available before conv_b on h_b=0
    // since conv_b at h_b=0 needs ya rows -1,0,1 (i.e. 0,1). We need ya[0] and ya[1].
    // Original code computed only ya[0] here. We need ya[0] AND ya[1] before conv_b loop,
    // because conv_b for h_b=0 needs ya[hh] for hh=-1,0,1 = only 0,1 in bounds.
    // Wait — original computes ya[0] here, then inside loop: simd_id>=16 lane computes
    // ya[h_b+2]. So for h_b=0: ya[2] gets computed in parallel with conv_b at h_b=0
    // which only needs ya[-1,0,1]. ya[1] is missing — but original works because the
    // pipeline pre-staged it differently? Let me recheck — original prelude only
    // computes ya at hh_=0 (group 0). For h_b=0 conv_b needs ya[0],ya[1]. ya[1] must
    // also be precomputed somewhere... looking again: original loop body has simd_id>=16
    // compute ya[h_b+2]. At iter h_b=0 conv_b reads ya[0,1], simd_id>=16 writes ya[2].
    // But ya[1] is never written! Unless... ya[1] was set by simd_groups simd_id=16..31
    // in the prelude? No — prelude uses `group = simd_id / C1` and `hh_=int(group)`,
    // so simd_id 0..15 -> group=0 hh_=0; simd_id 16..31 -> group=1 hh_=1. Yes! ya[1].
    // So prelude writes ya[0] AND ya[1]. Good. Keep the same logic.
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
                for (uint ci = 0; ci < C1; ++ci) {
                    s += float(h0p[ci]) * float(wp[ci]);
                }
            }
        }
        float v = fmax(s, 0.0f);
        ya_rows[(uint(hh_) & 3u) * WP * C1 + (w_ + 1u) * C1 + c] = half(v);
        ya_f[(uint(hh_) & 3u) * WP * C1 + (w_ + 1u) * C1 + c] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-row loop: conv_b via MMA on simd_id 0..3, conv_a on simd_id >= 16 (16..31)
    // for prefetch ya[h_b+2]. simd_id 4..15 stay idle on conv_b but help with conv_a?
    // We'll keep conv_a using simd_id 16..31 (16 simdgroups). The 16 simdgroups 0..15
    // do conv_b. Of those, only 4 are needed for MMA (4 w-tiles × 8 wide = 32). The
    // other 12 also need work. We'll split: simdgroups 0..3 do MMA conv_b; simdgroups
    // 4..15 do residual+GAP accumulation in parallel.
    //
    // Actually simpler: do MMA conv_b on simd 0..3, then have those same 4 simdgroups
    // also handle residual+ReLU+GAP write-back. Other simd 4..15 idle.
    //
    // Output for each conv_b tile (8 spatial w × 16 c_out):
    //   acc[2 N-tiles] of simdgroup_matrix<float,8,8>, indexed by c_out_outer (0..1)
    // For each (kh, kw) (9 taps):
    //   Load A: 8x8 tile of ya_f[slot_h, w_base+kw .. w_base+kw+7, c_in_outer*8..+7]
    //           for c_in_outer in {0,1}
    //   Load B: 8x8 tile of W_b[kh, kw, c_in_outer*8..+7, c_out_outer*8..+7]
    //   acc[c_out_outer] += A[c_in_outer] @ B[c_in_outer, c_out_outer]

    for (uint h_b = 0; h_b < H; ++h_b) {
        if (simd_id < 4u) {
            uint w_base = simd_id * 8u;  // 0, 8, 16, 24
            simdgroup_matrix<float, 8, 8> acc0 = simdgroup_matrix<float,8,8>(0.0f);
            simdgroup_matrix<float, 8, 8> acc1 = simdgroup_matrix<float,8,8>(0.0f);
            int hh_ = int(h_b);
            for (int kh = -1; kh <= 1; ++kh) {
                int hh = hh_ + kh;
                if (hh < 0 || hh >= 32) continue;
                uint slot_h = uint(hh) & 3u;
                for (int kw = -1; kw <= 1; ++kw) {
                    // A tiles: ya_f row at slot_h, 8 spatial positions (w_base+1+kw..+8+kw)
                    // ya_f layout: [slot, w (WP=34), c (C1=16)] row-major
                    threadgroup const float* a_ptr = &ya_f[slot_h * WP * C1 + (w_base + 1u + uint(kw)) * C1];
                    // Each "row" of A in matrix sense is one spatial pos (16 floats). Stride between rows = C1 = 16.
                    simdgroup_matrix<float, 8, 8> A0, A1;
                    simdgroup_load(A0, a_ptr,         /*stride=*/ C1, /*origin=*/ulong2(0, 0));
                    simdgroup_load(A1, a_ptr,         /*stride=*/ C1, /*origin=*/ulong2(8, 0));
                    // B tiles: W_b_tg[kh, kw, c_in, c_out]. Original W_b layout from host?
                    // host layout: W_b[c_out, kh, kw, c_in] (typical PyTorch OHWI->OIHW?).
                    // Looking at original kernel: wp = &W_a_tg[((c * 3 + kh+1) * 3 + kw+1) * C1];
                    // So index = ((c_out * 3 + kh+1) * 3 + kw+1) * C1 + c_in
                    //   -> layout [c_out, kh, kw, c_in], stride between c_out = 9*16=144
                    // For MMA we need contiguous [c_in, c_out] tile. So reload W_b into
                    // a (kh, kw)-grouped layout once. We did W_b_tg = W_b[i] above (same
                    // layout as device). We need [kh, kw, c_in, c_out] for MMA loads.
                    // Build a separate weight tile pointer on the fly: for each (kh,kw)
                    // we'd want B[c_in_outer*8+k, c_out_outer*8+n] from W_b[c_out, kh, kw, c_in].
                    // That's a transpose. We can use simdgroup_load with stride to transpose:
                    // For B with c_in as the "row" and c_out as the "column":
                    //   B[c_in, c_out] = W_b_tg[((c_out * 3 + kh+1) * 3 + kw+1) * C1 + c_in]
                    // For fixed (kh, kw), stride from c_in -> c_in+1 = 1 (contiguous)
                    // stride from c_out -> c_out+1 = 9*16 = 144
                    // simdgroup_load expects: load(matrix, ptr, elements_per_row_stride, origin)
                    // It loads matrix[m, n] from ptr[origin.y * stride + origin.x + (m * stride + n)]?
                    // Actually metal simdgroup_load: matrix M (M rows × N cols) loaded from
                    // memory where rows are 'stride' apart. So if we want B[c_in, c_out]
                    // where c_in is the row, we need rows of c_in to be 'stride' apart in mem.
                    // But c_in is contiguous (stride 1) and c_out has stride 144.
                    // We need to swap: load such that the "row" is c_out and the "col" is c_in,
                    // then use transpose via origin? Simpler: use the transpose-load form.
                    // Actually simdgroup_load has a 'transpose' parameter (bool).
                    //
                    // We'll load B with the natural layout: rows = c_out (stride 144),
                    // cols = c_in (stride 1) — then mark transpose=true so MMA sees rows=c_in.
                    threadgroup const float* wb_base = &W_b_tg[((0u * 3u + uint(kh+1)) * 3u + uint(kw+1)) * C1];
                    // For c_out_outer = 0 (c_out 0..7), c_in_outer = 0 (c_in 0..7):
                    //   start offset c_out=0, c_in=0 -> wb_base + 0
                    // For c_out 0..7, c_in 8..15: wb_base + 8 (c_in offset)
                    // For c_out 8..15, c_in 0..7: wb_base + 8 * 144
                    // For c_out 8..15, c_in 8..15: wb_base + 8 * 144 + 8
                    simdgroup_matrix<float, 8, 8> B00, B01, B10, B11;
                    // Stride between c_out rows = 9 * C1 = 144
                    simdgroup_load(B00, wb_base,            144, ulong2(0, 0), /*transpose=*/true);
                    simdgroup_load(B01, wb_base,            144, ulong2(8, 0), /*transpose=*/true);
                    simdgroup_load(B10, wb_base,            144, ulong2(0, 8), /*transpose=*/true);
                    simdgroup_load(B11, wb_base,            144, ulong2(8, 8), /*transpose=*/true);
                    // origin=ulong2(x, y) in simdgroup_load corresponds to (column, row) of
                    // the matrix being loaded. We want B's row = c_in, col = c_out.
                    // After transpose=true, the mem layout (row=c_out, col=c_in) becomes
                    // matrix (row=c_in, col=c_out). origin selects sub-tile.
                    //
                    // acc0 (c_out 0..7) += A0 @ B00 + A1 @ B10  (c_in 0..7, then 8..15)
                    // acc1 (c_out 8..15) += A0 @ B01 + A1 @ B11
                    simdgroup_multiply_accumulate(acc0, A0, B00, acc0);
                    simdgroup_multiply_accumulate(acc0, A1, B10, acc0);
                    simdgroup_multiply_accumulate(acc1, A0, B01, acc1);
                    simdgroup_multiply_accumulate(acc1, A1, B11, acc1);
                }
            }
            // Store acc to a temporary TG buffer, then add residual & ReLU & GAP accumulate
            // via the same 32 lanes (8 spatial × 4 lanes-per-pos? No, 32 lanes naturally
            // map to 8x4 — not enough for 8x16). Simpler: store to TG buf [8 w × 16 c],
            // then have all 32 lanes of this simdgroup do residual+ReLU+row_sum.
            threadgroup float tile[8 * C1];
            simdgroup_store(acc0, tile,     /*stride=*/C1, ulong2(0, 0));
            simdgroup_store(acc1, tile,     /*stride=*/C1, ulong2(8, 0));
            simdgroup_barrier(mem_flags::mem_threadgroup);
            // Now apply residual + ReLU + accumulate GAP. 8 w-pos × 16 c = 128 elements.
            // 32 lanes, so 4 elements each.
            for (uint elem = simd_lane; elem < 8u * C1; elem += 32u) {
                uint w_off = elem / C1;
                uint c     = elem % C1;
                uint w_    = w_base + w_off;
                float s = tile[elem];
                s += float(h0_rows[(h_b % 6u) * WP * C1 + (w_ + 1u) * C1 + c]);
                s = fmax(s, 0.0f);
                tile[elem] = s;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            // GAP: each lane contribute to gap[c]. Sum over 8 w-positions × 16 c.
            // Use lane = c (0..15) to accumulate column sum, then atomic? Simpler:
            // lane 0..15 each owns one c, sum 8 spatial positions, then atomic_add to gap[c].
            if (simd_lane < C1) {
                uint c = simd_lane;
                float colsum = 0.0f;
                for (uint w_off = 0; w_off < 8u; ++w_off) {
                    colsum += tile[w_off * C1 + c];
                }
                // Atomic add to gap[c] across simdgroups 0..3
                // No atomics on float in TG without atomic_float. Use threadgroup atomic.
                // Actually gap is small (16 floats). We have 4 simdgroups writing. Use
                // a per-simdgroup partial buffer then reduce.
                // Simpler: write to a [4 simdgroups × 16] partial buf, reduce later.
                // But that requires extra TG memory across iterations. Easiest: serialize
                // via atomic_fetch_add on atomic_float. Metal supports atomic_float in TG.
                // Fall back: write to partial[simd_id][c], reduce at end of iteration.
                // We'll use a dedicated buffer.
                // ---- using simple serial approach: only simdgroup 0 writes gap, others
                // store to TG partial and simdgroup 0 reduces. ----
                // For simplicity: just have each simdgroup write to gap_partial[simd_id * 16 + c],
                // then a single barrier and simdgroup 0 reduces.
                // But this needs the buffer declared outside. Add it.
            }
        }
        // conv_a prefetch on simd_id 16..31 (group 1 of 16): writes ya[h_b+2]
        if (simd_id >= 16u) {
            uint c  = simd_id - 16u;
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
                        for (uint ci = 0; ci < C1; ++ci) {
                            s += float(h0p[ci]) * float(wp[ci]);
                        }
                    }
                }
                float v = fmax(s, 0.0f);
                ya_rows[(uint(hh_) & 3u) * WP * C1 + (w_ + 1u) * C1 + c] = half(v);
                ya_f[(uint(hh_) & 3u) * WP * C1 + (w_ + 1u) * C1 + c] = v;
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
