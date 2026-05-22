// llama_decoder_layer (M4) — Redesigned. Single TG of 1024 threads.
// S=64 D=128 H=4 HKV=2 Dh=32 FF=256.
//
// Phase A: attention. Uses TG K[HKV,S,DH] + V[HKV,S,DH] + Q_buf + P_buf.
// Phase C: SwiGLU FFN tiled over rows. Per row-tile (8 rows): RMSNorm, compute
//          gu fused into fh = silu(gate)*up, then y += fh @ W_down.

#include <metal_stdlib>
using namespace metal;

#define S    64
#define D    128
#define H    4
#define HKV  2
#define DH   32
#define G    2
#define FF   256
#define TFF  512
#define TG   1024
#define QKV_W (D + 2*HKV*DH)

kernel void llama_decoder_layer_f32(
    device const float* x       [[buffer(0)]],
    device const float* W_qkv   [[buffer(1)]],
    device const float* W_o     [[buffer(2)]],
    device const float* W_gu    [[buffer(3)]],
    device const float* W_down  [[buffer(4)]],
    device       float* y       [[buffer(5)]],
    constant uint& S_   [[buffer(6)]],
    constant uint& D_   [[buffer(7)]],
    constant uint& H_   [[buffer(8)]],
    constant uint& HKV_ [[buffer(9)]],
    constant uint& FF_  [[buffer(10)]],
    constant float& base [[buffer(11)]],
    constant float& eps  [[buffer(12)]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    const uint t = tid.x;

    // === TG memory: 8192 floats = 32 KB total ===
    // Phase A layout:  K[4096] | V[4096]                          (32 KB)
    // We also need transient Q_buf[2048] + P_buf[4096] = 24 KB,
    // but we can re-use the LATER half of memory: place Q_buf at K (4096..6144)? No, K is needed.
    // So we declare a separate small TG pool for attention working set.
    //
    // To minimize TG mem, we put K & V in one pool (32 KB total), and Q_buf/P_buf in a SEPARATE TG
    // declaration that physically follows but Metal sums them. Metal TG limit is 32 KB total.
    // Trick: structure the kernel so that K+V live in pool A (32 KB) but Q_buf is reused across heads
    // and we avoid Q_buf entirely by computing scores directly into P_buf using just K and an on-the-fly Q.

    // Single 32-KB TG pool, aliased for different phases.
    threadgroup float pool[8192];        // 32 KB total
    threadgroup float* K = pool;                  // [HKV*S*DH] = 4096
    threadgroup float* V = pool + HKV*S*DH;       // [HKV*S*DH] = 4096
    // K + V = 32 KB. Anything else would exceed.

    // We compute attention but do not need Q across phases: we'll compute scores by streaming Q.
    // To do softmax we need scores in TG. We'll use the y device buffer as scratch (rows beyond S*D
    // don't exist — but the y output buffer is exactly S*D in size). So we cannot use y for scratch.
    //
    // Solution: compute Q_h and scores in two separate TG allocations *reusing* K's storage AFTER
    // we've used K for that head. But K is needed across all heads.
    //
    // Alt solution: split TG memory. Use K[4096] (16 KB) + scratch[4096] (16 KB). Drop V from TG.
    // V is then recomputed every time we use it in attn_out step. V recompute cost per head:
    //   per (kv, kt, dh) value = D mac = 128. Number = HKV*S*DH = 4096. Total per kernel call = 4096*128 = 524K macs.
    //   But this is per HEAD (H=4) since each head visits attn@V → so 4*524K = 2.1M. Negligible.
    //
    // We'll keep both K and V (clean) and *redesign attention to do soft-max with on-the-fly Q*:
    //   For each head, for each (sq, kt): compute Q[sq,:] on the fly (D macs × DH), then dot with K[kt,:].
    //   That's S*S*(D*DH + DH) = 64*64*(128*32+32) ~ 17M macs per head, *4 heads = 67M. Too slow.
    //
    // Best: stick with current attention scheme but tighten TG memory. Move Q_buf and P_buf into TG
    // but make them smaller / share. Total need: K(16K)+V(16K)=32K. Already at limit.
    //
    // The cleanest path: drop V from TG and compute V on the fly during attn@V step *per head*.
    // Free 16 KB for Q_buf(8 KB) + P_buf(16 KB)? That's 24 KB needing. Use 16 KB for P_buf only and
    // compute Q on the fly during P_buf build (one big matmul per head).

    // We'll go with the approach: keep K (16 KB) in TG. Use a second 16 KB TG region as "sc".
    //   - sc holds P_buf[4096] (S*S = 4096 floats) for one head.
    //   - V is recomputed inline when needed (small).

    // Phase A executes attention. We track rstd_x for each row in a small TG array.
    // Precompute rstd_x[S] into the LAST 64 floats of V (alias). V will be written later;
    // we copy rstd_x out to registers in each thread before V is written.
    threadgroup float* rstd_x_tg = V + (HKV*S*DH - S);
    if (t < S) {
        float ss = 0.0f;
        uint xoff = t * D;
        for (uint d = 0; d < D; ++d) { float v = x[xoff + d]; ss = fma(v, v, ss); }
        rstd_x_tg[t] = rsqrt(ss / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Build K[HKV,S,DH] with RoPE. K doesn't share storage with rstd_x_tg (which lives in V).
    for (uint idx = t; idx < HKV * S * DH; idx += TG) {
        uint kv = idx / (S * DH);
        uint rem = idx - kv * S * DH;
        uint s = rem / DH;
        uint dh = rem - s * DH;
        uint i = dh >> 1;
        uint pair_off = dh & 1u;
        uint colK0 = D + kv * DH + (i << 1);
        uint colK1 = colK0 + 1u;
        float rstd = rstd_x_tg[s];
        uint xoff = s * D;
        float k0v = 0.0f, k1v = 0.0f;
        for (uint d = 0; d < D; ++d) {
            float h_ = x[xoff + d] * rstd;
            k0v = fma(h_, W_qkv[d * QKV_W + colK0], k0v);
            k1v = fma(h_, W_qkv[d * QKV_W + colK1], k1v);
        }
        float omega = precise::exp(-(2.0f * float(i) / float(DH)) * log(base));
        float ang = float(s) * omega;
        float cv = cos(ang);
        float sv = sin(ang);
        float rv = (pair_off == 0u) ? (k0v * cv - k1v * sv) : (k0v * sv + k1v * cv);
        K[idx] = rv;
    }
    // We need rstd_x for V-build, but V-build overwrites the V[] storage where rstd_x lives.
    // Solution: each thread first loads ALL needed rstd_x values into private registers, then barrier,
    // then writes V (since each thread t handles up to 4 strided idx values, covering up to 4 distinct s).
    // Number of idx per thread: HKV*S*DH/TG = 4096/1024 = 4. So 4 s-values per thread.
    float rstd_loc[4];
    {
        // Determine s for each of this thread's 4 idx values: idx = t + step*TG, step=0..3.
        for (uint step = 0; step < 4; ++step) {
            uint idx = t + step * TG;
            uint kv = idx / (S * DH);
            uint rem = idx - kv * S * DH;
            uint s = rem / DH;
            rstd_loc[step] = rstd_x_tg[s];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Build V[HKV,S,DH]. Now safe to overwrite V (rstd_x stashed in registers).
    {
        for (uint step = 0; step < 4; ++step) {
            uint idx = t + step * TG;
            uint kv = idx / (S * DH);
            uint rem = idx - kv * S * DH;
            uint s = rem / DH;
            uint dh = rem - s * DH;
            uint colV = D + HKV * DH + kv * DH + dh;
            float rstd = rstd_loc[step];
            uint xoff = s * D;
            float vv = 0.0f;
            for (uint d = 0; d < D; ++d) {
                vv = fma(x[xoff + d] * rstd, W_qkv[d * QKV_W + colV], vv);
            }
            V[idx] = vv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Initialize y = x (residual base). 8192 cells / 1024 threads = 8 per thread.
    for (uint i = 0; i < 8; ++i) {
        uint cell = t * 8 + i;
        y[cell] = x[cell];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // Cache rstd_x for Q-build s-values per thread (2 values per thread).
    // BUT rstd_x_tg was overwritten by V-build. We could have saved it BEFORE V-build into another register.
    // We did save 4 values for V-build into rstd_loc; reuse those mapped to Q-build's needed s.
    // For Q-build, each thread t handles 2 idx values: t*2+0 and t*2+1, mapping to s = idx/DH.
    // V-build mapping: idx_v = t + step*TG. The s values seen by V-build differ from Q-build.
    // Cleanest: save rstd_x to a 2-element private array before V-build.
    // We already have rstd_loc[4] from V-build; let's also compute Q-build's rstd values from rstd_x_tg
    // BEFORE we corrupted V — but we did already. Need to redo.
    // To avoid restructure: just RECOMPUTE rstd locally now (after V-build) into a 2-element register array.
    float rstd_q[2];
    for (uint i = 0; i < 2; ++i) {
        uint idx = t * 2 + i;
        uint s = idx / DH;
        // inline RMSNorm of x[s,:]
        float ss = 0.0f;
        uint xoff = s * D;
        for (uint d = 0; d < D; ++d) { float v = x[xoff + d]; ss = fma(v, v, ss); }
        rstd_q[i] = rsqrt(ss / float(D) + eps);
    }

    // Phase A continues: per-head Q, scores, softmax, attn@V, then y += attn @ W_o.
    //
    // We can't allocate Q_buf + P_buf in TG (would exceed 32 KB after K+V).
    // So we shrink: drop V from TG (it's gone — but we still need it!). We'll keep V in TG, K in TG,
    // and use device buffer y[S*D..] — but y is only S*D long. So we MUST find another way.
    //
    // Solution: use a third TG buffer `sc` of just P_buf size. Since K (4096) + V (4096) = 8192 already
    // exhausts 32KB, we drop V. Instead, we run an *alternative*: after computing K, write V to a
    // temporary in TG that REPLACES Q after Q is consumed. But Q won't exist yet.
    //
    // Workaround: shrink V to HKV*S*DH = 4096 floats but reuse K's storage for V after attention loop
    // uses K for that head's scores AND probs in a single fused pass.
    //
    // Concretely: per head h:
    //   build Q_buf[S*DH] (replaces V usage temporarily? no, V is needed simultaneously).
    //
    // Easiest correct approach (used in current m4.metal): use the *output* y buffer as device scratch.
    // The harness gives a writable output of size S*D = 8192 floats. We can't write past it. But we can
    // overwrite y[0..S*D) and restore later. Actually we need y itself as accumulator.
    //
    // Cleanest: declare both K and V as TG, AND declare Q_buf + P_buf as TG. Total need:
    //   4096 + 4096 + 2048 + 4096 = 14336 floats = 56 KB. EXCEEDS 32 KB limit.
    //
    // So we DROP V from TG. We recompute V inside the attn@V step:
    //   attn_out[sq, dh] = sum_kt P[sq,kt] * V[kv, kt, dh]
    //   V[kv,kt,dh] = sum_d (x[kt,d]*rstd_x[kt]) * W_qkv[d, D+HKV*DH+kv*DH+dh]
    //   per (sq, dh), and we vary kt — this naive recompute is 64*DH*(D+S) per head = too slow.
    // We can pre-stage V into a SEPARATE TG buffer that overlaps with Q_buf+P_buf in time, not space.
    //
    // Better trick: declare TG memory layout dynamically. K[4096] + V[4096] = 32 KB.
    // For Q_buf & P_buf, store them INSIDE K's storage at the times when K's data is consumed.
    // But scores need both K and storage for output. Can't reuse.
    //
    // Final decision: KEEP K (16 KB) + ATTN scratch (16 KB). DROP V from TG; precompute V into
    // device y? No — y is the output. Use a DIFFERENT staging: re-write V into the K storage AFTER
    // scores are computed. That works because scores no longer need K (only P_buf, V).
    //
    // Plan per head:
    //   1. Build Q_buf[S*DH] in TG attn_sc[0..2048).
    //   2. Compute scores P_buf in attn_sc[2048..6144). Need 4096 floats — only 2048 left if scratch is 16KB (4096 floats).
    //      Resolution: attn_sc needs 6144 floats (Q + P) = 24 KB. K (16 KB) + 24 KB = 40 KB. Too big.
    //
    // OK — change tack. Use the K buffer's space for Q after attention dot products done with it.
    // We still need K and V simultaneously during scores (K) and attn@V (V) — but not at the same time.
    //
    // Plan with two TG buffers A & B of 16 KB each:
    //   A holds K (built once, used for scores), then OVERWRITTEN to hold Q+P or P? No — K used per head.
    //   Actually K stays the same across heads. We just need it for scores in each head.
    //
    // We'll have TG memory: K[4096] (16 KB), V[4096] (16 KB) = 32 KB. Per-head Q/P/attn_out done in REGISTERS
    // by distributing work.
    //
    // For each head:
    //   - Score block S*S = 4096; with 1024 threads, 4 scores per thread. Each thread computes its 4 scores
    //     completely with no intermediate Q_buf: it loads Q on the fly (Q[sq,d] = sum_e h_x[sq,e]*W_qkv[e,colQ]).
    //     But Q must be RoPE'd and computed from x. Computing Q on demand per score is 4 * (DH*(D + small)) per
    //     thread = 4*32*128 ≈ 16K macs/thread for Q only, ×1024 = 16M macs just to build Q across scores.
    //     But Q used DH times. We'd recompute Q DH times = 32× redundancy → S*S*DH*D = 64*64*32*128 = 16.8M
    //     macs total = ~ negligible actually.
    //
    // Better: split Q computation so that each thread caches its share of Q across its 4 scores.
    // With 4 scores per thread (sq fixed, varying kt within block? need a layout where same sq):
    //   Layout: thread t handles 4 scores (sq, kt0..kt3) with same sq. Q[sq,:] computed once and reused.
    //   1024 threads, S*S/4 = 1024 score blocks where each block = (sq, 4 kts). 64 sq × 16 (kt blocks of 4) = 1024.
    //
    // Q[sq,dh] for all dh requires loading sq's row of x (rstd) and full W_qkv col DH. 32 MACs of D = 4K. Each
    // thread does this once (cache Q row in registers — 32 floats), then 4 dot products of length DH = 128 macs.
    // Total per head: 1024 * (4K + 128) ≈ 4.2M macs. ×4 heads = 16.8M. Big but feasible.
    //
    // But softmax needs all scores for a row → after computing scores, we still need them in TG to softmax.
    // We'd write to P_buf in TG (S*S=4096 = 16 KB). But K + V already occupy 32 KB. Add 16 KB → 48 KB. Too big.

    // OK — drop V from TG. Keep K[4096]=16KB and P_buf[4096]=16KB. Recompute V where needed:
    //   attn_out[sq, dh] = sum_kt P[sq,kt] * V[kv, kt, dh]
    //   We can recompute V[kv,kt,dh] on the fly. Per attn_out entry, we sum over kt, each kt requires
    //   reading V[kv,kt,dh] which is sum_d (x[kt,d]*rstd_x[kt]) * W_qkv[d, colV]. Recomputing inside
    //   the kt loop is 64 × 128 macs per (sq, dh) → 64*64*32 cells × 128 macs = 16.8M macs / head ×4 = 67M.
    //   Way too slow.
    //
    // Better: precompute V into a temporary, but only one slice at a time. Or: change order of attn@V:
    //   out[sq,dh] = sum_kt P[sq,kt] * V[kt,dh].
    //   Move the sum_d innermost: V[kt,dh] = sum_d (x[kt,d]*rstd) * W_qkv[d, colV].
    //   Reorder: out[sq,dh] = sum_d ( sum_kt P[sq,kt] * x[kt,d]*rstd_x[kt] ) * W_qkv[d, colV].
    //   Let A[sq, d] = sum_kt P[sq,kt] * x[kt,d]*rstd_x[kt]. Then out[sq,dh] = sum_d A[sq,d] * W_qkv[d, colV].
    //   A is (S, D) = 8192 floats = 32 KB. Even bigger than V.
    //
    // OK best to just put V back in TG. With K+V = 32 KB we're maxed.

    // Strategy: use K+V (32 KB) + a smaller P_buf overlapping with K reuse.
    // P_buf[4096] would need 16 KB. We can split K into two TG regions K_a[2048]+K_b[2048] (HKV slices, each 8 KB).
    // K_a = K[0], K_b = K[1]. We compute scores for head h using K[kv]; after computing scores for head h,
    // we no longer need K[kv]. But scores must persist into softmax+attn_out, which still needs V.
    //
    // Tweak: P_buf[4096] alias = upper half of V (the unused kv slice for this head). Each head uses
    // only one kv slice (kv = h_/G, G=2 → kv=0 for heads 0,1; kv=1 for heads 2,3). So when processing head 0,
    // V[kv=1] is unused → use V[kv=1] region as P_buf. Same for heads 1 (kv=0 used, kv=1 free).
    // For heads 2,3 (kv=1 used, kv=0 free).
    //
    // V slice is S*DH = 2048 floats = 8 KB. But P_buf needs 4096 floats = 16 KB. Doesn't fit in one V slice.
    //
    // We could use the K slice not in use: same problem — one slice is 2048 floats.
    //
    // What about combining unused K and V slices? unused K_kv + unused V_kv = 4096 floats = 16 KB. Yes! Use this.
    //
    // For each head h with kv = h_/G:
    //   used: K[kv], V[kv]
    //   unused (in terms of data values needed): K[1-kv], V[1-kv]
    //   But K and V for *both* kv slices are still needed across ALL heads. We can't overwrite the unused slice
    //   in the current head because the NEXT head may need it.
    //
    // Order heads so that head 0 uses kv=0, head 1 uses kv=0 (kv=1 still needed for heads 2,3). So we can't
    // overwrite K[1]/V[1] during heads 0/1.
    //
    // Compromise: only overwrite the slices when last used:
    //   Heads sequence by kv: heads {0,1} kv=0, heads {2,3} kv=1.
    //   During heads 0,1 we have K[1]+V[1] as the unused-but-needed-later slice → can't use as scratch.
    //   During heads 2,3, we have K[0]+V[0] still potentially needed only by past heads → unused now → use as scratch.
    //
    // So only the last 2 heads benefit. Marginal.
    //
    // Approach decision: just allocate 32 KB TG = K+V, and use the harness's y output buffer as device scratch
    // for P_buf temporarily (y is 8192 floats; we can use part of it during attention, then re-populate y
    // before residual sum). Specifically, since y is only S*D=8192 floats and we need P_buf=4096 floats,
    // we can store scores in y[0..4096). But y must hold the residual base x → keep residual in REGISTERS:
    //   each thread holds 8 elements of x in registers as y_acc (this is what the original code did).
    //   y device buffer is used purely as scratch during attention. After attention we write y_acc + attn to y.
    //
    // This works. Let's commit to it.

    // Per-thread register accumulator for output (residual + attn).
    float y_acc[8];
    for (uint i = 0; i < 8; ++i) {
        uint cell = t * 8 + i;
        y_acc[i] = x[cell];
    }

    const float inv_sqrt_dh = rsqrt(float(DH));

    for (uint h_ = 0; h_ < H; ++h_) {
        const uint kv = h_ / G;

        // (a) compute Q_h and write to y[0 .. S*DH) as scratch. S*DH = 2048 floats.
        // 2048 / 1024 = 2 per thread.
        for (uint i = 0; i < 2; ++i) {
            uint idx = t * 2 + i;
            uint s = idx / DH;
            uint dh = idx - s * DH;
            uint qi = dh >> 1;
            uint pair_off = dh & 1u;
            uint colQ0 = h_ * DH + (qi << 1);
            uint colQ1 = colQ0 + 1u;
            float rstd = rstd_q[i];
            uint xoff = s * D;
            float q0v = 0.0f, q1v = 0.0f;
            for (uint d = 0; d < D; ++d) {
                float h_v = x[xoff + d] * rstd;
                q0v = fma(h_v, W_qkv[d * QKV_W + colQ0], q0v);
                q1v = fma(h_v, W_qkv[d * QKV_W + colQ1], q1v);
            }
            float omega = precise::exp(-(2.0f * float(qi) / float(DH)) * log(base));
            float ang = float(s) * omega;
            float cv = cos(ang);
            float sv = sin(ang);
            float rv = (pair_off == 0u) ? (q0v * cv - q1v * sv) : (q0v * sv + q1v * cv);
            y[idx] = rv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // (b) compute scores P[sq,kt] = Q[sq,:] · K[kv,kt,:] * inv_sqrt_dh.
        // Store P in y[S*DH .. S*DH + S*S) = y[2048 .. 6144).
        // S*S = 4096 / 1024 = 4 per thread.
        for (uint i = 0; i < 4; ++i) {
            uint idx = t * 4 + i;
            uint sq = idx / S;
            uint kt = idx - sq * S;
            float dot = 0.0f;
            for (uint d = 0; d < DH; ++d) {
                dot = fma(y[sq * DH + d], K[kv * S * DH + kt * DH + d], dot);
            }
            y[2048u + idx] = dot * inv_sqrt_dh;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // (c) softmax over each row of P. 1024 threads / S=64 rows = 16 lanes/row.
        {
            uint sq = t >> 4;
            uint lane = t & 15u;
            uint base_off = 2048u + sq * S;
            float v0 = y[base_off + lane];
            float v1 = y[base_off + lane + 16];
            float v2 = y[base_off + lane + 32];
            float v3 = y[base_off + lane + 48];
            float m = max(max(v0, v1), max(v2, v3));
            m = max(m, simd_shuffle_xor(m, 1));
            m = max(m, simd_shuffle_xor(m, 2));
            m = max(m, simd_shuffle_xor(m, 4));
            m = max(m, simd_shuffle_xor(m, 8));
            v0 = precise::exp(v0 - m);
            v1 = precise::exp(v1 - m);
            v2 = precise::exp(v2 - m);
            v3 = precise::exp(v3 - m);
            float ss = v0 + v1 + v2 + v3;
            ss += simd_shuffle_xor(ss, 1);
            ss += simd_shuffle_xor(ss, 2);
            ss += simd_shuffle_xor(ss, 4);
            ss += simd_shuffle_xor(ss, 8);
            float inv = 1.0f / ss;
            y[base_off + lane]      = v0 * inv;
            y[base_off + lane + 16] = v1 * inv;
            y[base_off + lane + 32] = v2 * inv;
            y[base_off + lane + 48] = v3 * inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // (d) attn_out[sq, dh] = sum_kt P[sq,kt] * V[kv, kt, dh].
        // Store in y[6144 .. 6144 + S*DH = 8192). 2048 cells / 1024 = 2 per thread.
        for (uint i = 0; i < 2; ++i) {
            uint idx = t * 2 + i;
            uint sq = idx / DH;
            uint dh = idx - sq * DH;
            float acc = 0.0f;
            for (uint kt = 0; kt < S; ++kt) {
                acc = fma(y[2048u + sq * S + kt], V[kv * S * DH + kt * DH + dh], acc);
            }
            y[6144u + idx] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // (e) accumulate y_acc[s, d] += sum_dh attn_out[s, dh] * W_o[h_*DH+dh, d].
        for (uint i = 0; i < 8; ++i) {
            uint cell = t * 8 + i;
            uint s = cell / D;
            uint dout = cell - s * D;
            float acc = 0.0f;
            for (uint dh = 0; dh < DH; ++dh) {
                acc = fma(y[6144u + s * DH + dh], W_o[(h_ * DH + dh) * D + dout], acc);
            }
            y_acc[i] += acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }

    // Write y = y_acc.
    for (uint i = 0; i < 8; ++i) {
        y[t * 8 + i] = y_acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // ---------- Phase C: SwiGLU FFN, tiled over rows ----------
    // Reuse K[] and V[] (combined 8192 floats = 32 KB) as the FFN TG pool.
    // For each tile of ROWS_TILE rows:
    //   ln_block[ROWS_TILE * D]  (RMSNorm output)
    //   fh_block[ROWS_TILE * FF] (silu(gate)*up)
    //
    // ROWS_TILE = 8 → ln = 1024 floats = 4 KB, fh = 2048 floats = 8 KB, total = 12 KB. Fits.

    const uint ROWS_TILE = 8;
    threadgroup float* ln_block = pool;                                // [ROWS_TILE*D] = 1024
    threadgroup float* fh_block = pool + ROWS_TILE * D;                // [ROWS_TILE*FF] = 2048
    threadgroup float* rstd_y_tile_g = pool + ROWS_TILE * D + ROWS_TILE * FF;  // [ROWS_TILE]
    const uint N_TILES = S / ROWS_TILE;

    for (uint tile = 0; tile < N_TILES; ++tile) {
        uint row_base = tile * ROWS_TILE;
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // (1) RMSNorm ROWS_TILE=8 rows into ln_block. 8*128=1024 cells. 1 per thread.
        threadgroup float* rstd_y_tile = rstd_y_tile_g;
        {
            uint r = t >> 7;            // 0..7
            uint d = t & 127u;          // 0..127

            uint row = row_base + r;
            float v = y[row * D + d];
            float sq = v * v;

            float simd_part = simd_sum(sq);
            uint sg = d >> 5;
            uint lane = d & 31u;
            if (lane == 0) {
                ln_block[r * 4 + sg] = simd_part;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0 && lane == 0) {
                float total = ln_block[r * 4 + 0] + ln_block[r * 4 + 1] + ln_block[r * 4 + 2] + ln_block[r * 4 + 3];
                rstd_y_tile[r] = rsqrt(total / float(D) + eps);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float rstd = rstd_y_tile[r];
            ln_block[r * D + d] = v * rstd;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // (2) Compute gu fused into fh_block. 8*256 = 2048 outputs / 1024 = 2 per thread.
        for (uint i = 0; i < 2; ++i) {
            uint idx = t * 2 + i;
            uint r = idx / FF;
            uint f = idx - r * FF;
            float gate = 0.0f, up = 0.0f;
            uint ln_off = r * D;
            uint wg_g = f;
            uint wg_u = FF + f;
            for (uint d = 0; d < D; ++d) {
                float lv = ln_block[ln_off + d];
                gate = fma(lv, W_gu[d * TFF + wg_g], gate);
                up   = fma(lv, W_gu[d * TFF + wg_u], up);
            }
            float silu_g = gate / (1.0f + precise::exp(-gate));
            fh_block[r * FF + f] = silu_g * up;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // (3) y[row, d] += sum_f fh_block[r, f] * W_down[f, d].
        // 8*128 = 1024 cells / 1024 = 1 per thread.
        {
            uint r = t >> 7;
            uint d = t & 127u;
            uint row = row_base + r;
            float acc = 0.0f;
            uint fh_off = r * FF;
            for (uint f = 0; f < FF; ++f) {
                acc = fma(fh_block[fh_off + f], W_down[f * D + d], acc);
            }
            y[row * D + d] += acc;
        }
    }
}
