// transformer_block — pre-LN BERT/ViT block (M4 redesign)
// One TG of 1024 threads = 32 simdgroups. S=64 D=128 H=4 Dh=32 FF=256.
//
// Layout: each simdgroup handles one token-row at a time. Two passes (32 sg * 2 = 64 rows).
// Attention path uses the same 32-sg organization. FFN uses contiguous-f accumulation
// per lane to maximize coalesced reads into W_ff1 and W_ff2.
#include <metal_stdlib>
using namespace metal;

#define S  64
#define D  128
#define H  4
#define DH 32
#define FF 256
#define TG 1024

kernel void transformer_block_f32(
    device const float* x      [[buffer(0)]],
    device const float* W_qkv  [[buffer(1)]],
    device const float* W_o    [[buffer(2)]],
    device const float* W_ff1  [[buffer(3)]],
    device const float* W_ff2  [[buffer(4)]],
    device       float* y      [[buffer(5)]],
    constant uint& S_  [[buffer(6)]],
    constant uint& D_  [[buffer(7)]],
    constant uint& H_  [[buffer(8)]],
    constant uint& FF_ [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    const uint t    = tid.x;
    const uint sg   = t >> 5;     // 0..31
    const uint lane = t & 31;     // 0..31

    // TG memory regions (overlap heavily; reused across phases)
    // Phase A (attention): need ln1[S*D=8192], Q/K/V tiles [S*DH=2048 each = 6144], scores [S*S=4096]
    // Phase B (FFN): need y_resid[S*D=8192] OR ln2[S*D=8192], hidden[FF*S? no, per-row 256]
    // We'll use 32KB total: 8192 floats.
    threadgroup float pool[8192];

    // ============================================================
    // PHASE A: Attention
    // ============================================================
    // Step 1: Compute LN1(x) into pool[0..S*D]. Each sg handles 2 rows.
    {
        threadgroup float* LN = pool;
        // 32 sg, S=64 rows -> 2 rows per sg
        for (uint pass = 0; pass < 2; ++pass) {
            uint s = sg + pass * 32;
            uint off = s * D;
            // lane reads 4 contiguous d's: lane*4..lane*4+3
            uint d0 = lane * 4;
            float v0 = x[off + d0 + 0];
            float v1 = x[off + d0 + 1];
            float v2 = x[off + d0 + 2];
            float v3 = x[off + d0 + 3];
            float s1 = v0 + v1 + v2 + v3;
            float s2 = v0*v0 + v1*v1 + v2*v2 + v3*v3;
            s1 = simd_sum(s1);
            s2 = simd_sum(s2);
            float m = s1 / float(D);
            float var = max(s2/float(D) - m*m, 0.0f);
            float rs = rsqrt(var + eps);
            LN[off + d0 + 0] = (v0 - m) * rs;
            LN[off + d0 + 1] = (v1 - m) * rs;
            LN[off + d0 + 2] = (v2 - m) * rs;
            LN[off + d0 + 3] = (v3 - m) * rs;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: QKV projection + attention, per head.
    // We'll process all H heads. For each head we need Q,K,V each of shape (S, DH=32).
    // We'll store Q,K,V tiles after the LN region? But we need LN later for next head.
    // Use a separate region: pool offset 0 stays LN; tiles go at 8192+? No, we have only 8192.
    // Solution: copy ln out into a scratch space within pool? Reuse pool tail.
    // Actually pool is 8192 = S*D, perfectly filled by LN. We need another 6144 for Q,K,V tile + 4096 for scores.
    // We can't store all simultaneously. Strategy: keep LN, store Q,K,V,scores via scratch registers
    // by reusing parts of pool we don't need: scores can overlap with Q after we no longer need Q.
    //
    // Simpler: increase pool. Max TG mem is 32KB on M4 (== 8192 floats). We're at the limit.
    //
    // Alternative: for each head, write Q/K/V into a small region (S*DH*3 = 6144),
    // do attention computing output_h (S*DH = 2048), then directly accumulate into y (device).
    // But we still need LN access for next head's Q/K/V proj.
    //
    // LAYOUT:
    //   pool[0..8191] = LN1 (8192 floats) — kept across head loop
    // We need head-local scratch for Q,K,V,scores,out. We'll use a SECOND tg array.
    // M4 limit: 32KB. Let's reduce LN to register-resident across heads? Too much (8192 floats / 1024 thrs = 8 each — ok).
    //
    // Better plan: After computing LN, also reload LN values per-token per-head into registers.
    // But we need cross-token access (different sg's read different rows). Must use tg memory.
    //
    // Use registers per-lane to hold LN tile slices? Each sg handles 2 token rows, so 2*D=256 floats — too many regs.
    //
    // Accept the LN tg cost and use a second tg buffer for head scratch.
    // We can claim another array — total tg mem allocated is set by registry to 32KB. Let me check:
    // registry says threadgroup=(1024,1,1), no explicit tg mem? Metal will allocate what kernel declares.
    // M4 allows up to 32KB per TG. Let me allocate a second array.
    // Actually pool is 8192*4 = 32KB — already at limit. Cannot add more.
    //
    // ALTERNATIVE: overlap pool regions cleverly.
    //   pool[0..S*D=8192] = LN1  during QKV proj
    //   then we don't need LN1 after computing Q,K,V for that head (each head only needs LN once).
    //   So PER-HEAD: compute Q,K,V (which OVERWRITES LN1 progressively), then attention.
    //   But heads 1..3 need LN1 again. Recompute LN1 each head? Cheap (S*D = 8192 ops per head, 4 heads = 32K).
    //
    // Even simpler: compute LN1 once, copy x's residual already, then for each head:
    //   Q,K,V tiles in head-scratch, attention, accumulate into y[device].
    // We need TG memory for Q,K,V (6144), scores (4096) — that's 10240 > 8192. Doesn't fit alongside LN1.
    //
    // PLAN B: do not store LN1 in tg memory. Instead, recompute LN1 on the fly each time
    // we need a row. Each lane caches its 4-float slice of its 2 row LN values.
    //
    // Actually simplest: compute LN1 once, store in pool. For attention, since we don't need
    // it past QKV projection of the current head, overwrite half of pool with Q/K/V/scores.
    // LN1 = 8192. Q+K+V+scores tile = 6144+4096 = 10240. Doesn't fit if we keep LN1.
    //
    // Conclusion: recompute LN1 per head (cheap) and free pool for head scratch.
    // OR: store LN1 in a sticky place and process heads with smaller scratch.
    //
    // BEST: since DH=32 and S=64, each Q/K/V is 2048 floats. We can put:
    //   pool[0..2047]  = Q
    //   pool[2048..4095] = K
    //   pool[4096..6143] = V
    //   pool[6144....]   = scores (S*S=4096)... but 6144+4096 = 10240 > 8192.
    // We can reuse Q's region for scores after Q is no longer needed (scores = QK^T, then softmax,
    // then attn = softmax @ V — we need scores while attn_out being computed, no longer need Q).
    //   pool[0..4095] = scores (after computing QK^T overwriting Q region: needs 4096 floats; Q used 2048; need extra)
    //
    // Let me just allocate enough: declare pool as 10240? 10240*4 = 40KB > 32KB limit.
    //
    // OK redesign: don't store LN1 anywhere persistent. Compute LN1 stats per-row, cache (m, rs) per row
    // in a small tg array (S=64 entries * 2 = 128 floats). When projecting Q/K/V for head h,
    // each lane re-reads x[off+d] and applies (x-m)*rs, then matmuls. Reads x twice (once for stats,
    // once per head) — 4x extra reads but cheap.
    //
    // Wait — Q,K,V projections are matmul over D. Each (s, qd) output = sum_d LN[s,d] * W_qkv[d, qd].
    // We could fuse: each lane for row s holds LN[s,d_lane*4..d_lane*4+3] in registers across the whole head loop.
    // Per row, 4 floats in 32 lanes = 128 floats = D. Each sg owns 2 rows -> 8 floats per lane = 256 floats per lane.
    // That's a lot of regs but might work.
    //
    // I'll go with simpler: keep LN1 in pool[0..S*D]. Use a SEPARATE small head scratch.
    //
    // For attention head loop, the scratch needs:
    //   Q[S*DH=2048], K[S*DH=2048], V[S*DH=2048], scores[S*S=4096]
    // Reuse: compute Q, then compute K, then scores = Q@K^T -> we still need K and Q? No, after scores
    // we don't need Q. So reuse Q space for nothing here, but K and V still needed (V for attn_out).
    // Then softmax in place on scores. attn_out = scores @ V — need scores and V.
    //
    // Layout option:
    //   tile[0..2047]   = Q
    //   tile[2048..4095] = K
    //   tile[4096..6143] = V
    //   tile[6144..10239] = scores  — 10240 floats needed, can we afford 40KB? No, M4 max 32KB.
    //
    // Compress: use Q's slot for scores after Q is dead.
    //   tile[0..4095] = scores (after Q consumed)
    //   tile[2048..4095] = K (occupied until scores done)
    //   conflict: scores need 4096 contiguous floats but K is in [2048..4095].
    //   Use scores[0..S*S=4096] starting at offset 0, overwriting Q. But K at [2048..4095] would be
    //   overwritten by scores[2048..4095]. So move K to a different slot.
    //
    // Cleanest layout (10240 floats needed for full pre-overlap):
    //   K at [0..2047], V at [2048..4095], Q at [4096..6143], scores at [4096..8191] (overlap Q)
    //   total = 8192 floats = 32KB ✓
    //
    // BUT we also need LN1[S*D=8192] in pool simultaneously. That's another 32KB. Doesn't fit.
    //
    // FINAL plan: don't keep LN1 in tg memory. Cache means/rstds (S*2 = 128 floats).
    // Each lane stores its LN row slice in registers across the head loop.
    //
    // For Q,K,V projection: each sg handles 2 rows (s = sg, sg+32). For each row, lane owns
    // LN[s, lane*4..lane*4+3] = 4 floats. Across 2 rows = 8 floats per lane. Fine.
    //
    // Then Q,K,V tiles in tg memory (6144 floats).
    // scores in tg memory using overlap.
    //
    // Layout for tile region (after LN computed and discarded):
    //   pool[0..6143]    = Q,K,V (concatenated)
    //   pool[6144..8191] = (free, 2048 floats — but scores need 4096)
    // overlap: scores overwrites Q[0..4095]? Q is 2048 floats at [0..2047]. scores 4096 needs more.
    //   Put: Q[0..2047], K[2048..4095], V[4096..6143]. After scores=Q@K^T computed, Q is dead.
    //   scores[0..4095] writes into Q[0..2047] and K[2048..4095]. But we still need K for nothing
    //   after Q@K^T finishes (K consumed). So OK, overwrite K too.
    //   V remains at [4096..6143], scores at [0..4095], attn_out can go at... we have 2048 free at [6144..].
    //   attn_out[S*DH=2048] at [6144..8191] ✓
    //
    // After attn_out computed, we no longer need V or scores. Project attn_out @ W_o -> add to y[device].
    // ============================================================

    // Compute LN1 stats locally per (sg, pass) and keep in registers.
    // Each sg handles rows s = sg and sg+32 (pass 0,1).
    float ln1_m[2], ln1_rs[2];
    {
        for (uint pass = 0; pass < 2; ++pass) {
            uint s = sg + pass * 32;
            uint off = s * D;
            uint d0 = lane * 4;
            float v0 = x[off + d0 + 0];
            float v1 = x[off + d0 + 1];
            float v2 = x[off + d0 + 2];
            float v3 = x[off + d0 + 3];
            float s1 = v0 + v1 + v2 + v3;
            float s2 = v0*v0 + v1*v1 + v2*v2 + v3*v3;
            s1 = simd_sum(s1);
            s2 = simd_sum(s2);
            float m = s1 / float(D);
            float var = max(s2/float(D) - m*m, 0.0f);
            ln1_m[pass] = m;
            ln1_rs[pass] = rsqrt(var + eps);
        }
    }
    // Also copy x -> y (residual). We'll add attention output later.
    for (uint i = t; i < S*D; i += TG) y[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-head loop
    for (uint h = 0; h < H; ++h) {
        threadgroup float* Qt = pool + 0;       // [S*DH=2048]
        threadgroup float* Kt = pool + 2048;    // [S*DH=2048]
        threadgroup float* Vt = pool + 4096;    // [S*DH=2048]
        // 'attn_out' will be at pool+6144 [S*DH=2048]
        threadgroup float* Ot = pool + 6144;

        // Project Q,K,V for this head.
        // Output shape per head: (S, DH=32). Total 2048 outputs.
        // 32 sg = handle 2 rows each. For row s, each lane handles DH=32 columns -> 1 col per lane.
        // To compute Q[s, c] = sum_d LN[s,d] * W_qkv[d, h*DH + c]:
        //   each lane c reads W_qkv[d*(3D) + h*DH + c] for d=0..127 — D-strided access (stride 3D=384 floats).
        //   That's not great but only D=128 reads per output -> small.
        //
        // Better: cooperative — each sg handles 2 rows. For row s, broadcast LN[s,d] across the sg
        // and have lanes accumulate DH outputs. Need ln[s, d] per d.
        //
        // SIMD approach for row s:
        //   Step a: load LN[s, lane*4..lane*4+3] into 4 regs per lane (LN cooperatively recomputed).
        //   Step b: for each c in [0..DH=32], want sum_d LN[s,d] * W_qkv[d, h*DH+c].
        //     The 4 d-values held by each lane sum to one lane's partial; need simd_sum across lanes.
        //   Per (s,c) output: 32 lanes do 4 fmacs each, then simd_sum -> 1 output. We have 32 outputs (c=0..31).
        //   So 32 iters of c, 32 lanes each. Total 32*32*4 = 4096 fmacs to produce one row's Q.
        //   And we need 3 outputs (Q,K,V) per row -> 12288 fmacs per row. 2 rows per sg per head, 4 heads
        //   -> 12288*2*4 = 98304 fmacs per sg total. OK.
        //
        // Refactor: combine Q,K,V in one pass — for each d, lane reads LN[s,d] (already in reg),
        // accumulate into 3 partials per c. Or just do them separately.
        //
        // To make it efficient: for each row s in sg's pair:
        //   load 4 LN values into l0..l3
        //   each lane c (0..31) computes Q[c], K[c], V[c] simultaneously:
        //     qa = ka = va = 0
        //     for d=0..127: lane uses ln[s,d] (which lane owns it?) Need cross-lane.
        //
        // CLEANER: each lane owns output index c=lane. It loops d=0..127, reads ln[s,d] from a
        // broadcast/shuffle, reads W_qkv[d*3D + h*DH + c] = W_qkv[d*384 + h*32 + lane].
        //   This is one read per (d, lane) — coalesced! Lanes c=0..31 read 32 consecutive floats at d*384 + h*32.
        //   ln[s,d] needs to be broadcast.
        //
        // For broadcasting ln[s,d]: lane owns ln[s, lane*4..lane*4+3]. To access ln[s,d], use
        // simd_shuffle from lane = d/4, then pick element d%4. Use simd_broadcast.
        //
        // Let's code this up.

        // pass over 2 rows
        for (uint pass = 0; pass < 2; ++pass) {
            uint s = sg + pass * 32;
            uint off = s * D;
            float m = ln1_m[pass];
            float rs = ln1_rs[pass];
            uint d0 = lane * 4;
            float l0 = (x[off + d0 + 0] - m) * rs;
            float l1 = (x[off + d0 + 1] - m) * rs;
            float l2 = (x[off + d0 + 2] - m) * rs;
            float l3 = (x[off + d0 + 3] - m) * rs;

            float qa = 0, ka = 0, va = 0;
            const uint hoff = h * DH;
            // Block of 4 d's per outer iter: 4 shuffles total per block.
            for (uint blk = 0; blk < 32; ++blk) {
                float c0 = simd_shuffle(l0, blk);
                float c1 = simd_shuffle(l1, blk);
                float c2 = simd_shuffle(l2, blk);
                float c3 = simd_shuffle(l3, blk);
                uint base = blk * 4 * (3u*D) + hoff + lane;
                qa += c0 * W_qkv[base + 0*(3u*D)];
                ka += c0 * W_qkv[base + 0*(3u*D) + D];
                va += c0 * W_qkv[base + 0*(3u*D) + 2u*D];
                qa += c1 * W_qkv[base + 1*(3u*D)];
                ka += c1 * W_qkv[base + 1*(3u*D) + D];
                va += c1 * W_qkv[base + 1*(3u*D) + 2u*D];
                qa += c2 * W_qkv[base + 2*(3u*D)];
                ka += c2 * W_qkv[base + 2*(3u*D) + D];
                va += c2 * W_qkv[base + 2*(3u*D) + 2u*D];
                qa += c3 * W_qkv[base + 3*(3u*D)];
                ka += c3 * W_qkv[base + 3*(3u*D) + D];
                va += c3 * W_qkv[base + 3*(3u*D) + 2u*D];
            }
            Qt[s*DH + lane] = qa;
            Kt[s*DH + lane] = ka;
            Vt[s*DH + lane] = va;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step: compute scores = Q @ K^T (S x S) and softmax.
        // 32 sg * 32 lanes = 1024 threads, S*S = 4096 entries. Each thread does 4 entries.
        // Use storage in Ot region for now? Ot is at pool+6144 (2048 floats). Need 4096.
        // Reuse: Q is no longer needed individually after we have Q tile in tg. Actually we need Q
        // and K to compute scores. After scores done, Q is dead. So compute scores into a new buffer.
        // Put scores at pool[0..4095]? But that overwrites Q (needed during compute) and K.
        // After computing scores fully, we don't need Q,K anymore. Need V.
        // Use a temp: each thread holds 4 score values in regs, sync, then overwrite Q+K with scores.
        const float inv_sqrt_dh = rsqrt(float(DH));
        float sc_vals[4];
        for (uint i = 0; i < 4; ++i) {
            uint pair = t * 4 + i;
            uint sq = pair / S;
            uint kt = pair % S;
            float dot = 0;
            for (uint d = 0; d < DH; ++d) {
                dot += Qt[sq*DH + d] * Kt[kt*DH + d];
            }
            sc_vals[i] = dot * inv_sqrt_dh;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Write scores into pool[0..4095] (overwriting Q,K).
        threadgroup float* Sc = pool;
        for (uint i = 0; i < 4; ++i) {
            uint pair = t * 4 + i;
            Sc[pair] = sc_vals[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Softmax per row of S (64 rows). 32 sg -> 2 rows each. Each sg has 32 lanes, row has S=64 entries.
        // lane handles 2 entries.
        for (uint pass = 0; pass < 2; ++pass) {
            uint sq = sg + pass * 32;
            uint base = sq * S;
            float v0 = Sc[base + lane];
            float v1 = Sc[base + lane + 32];
            float mx_ = max(v0, v1);
            mx_ = simd_max(mx_);
            v0 = exp(v0 - mx_);
            v1 = exp(v1 - mx_);
            float ss = v0 + v1;
            ss = simd_sum(ss);
            float inv = 1.0f / ss;
            Sc[base + lane]      = v0 * inv;
            Sc[base + lane + 32] = v1 * inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // attn_out = softmax @ V  -> (S, DH=32). Store in Ot.
        // 32 sg * 32 lanes = 1024 threads, output has 2048 entries -> 2 per thread.
        // Better: each sg handles 2 rows, each lane handles 1 output column.
        for (uint pass = 0; pass < 2; ++pass) {
            uint sq = sg + pass * 32;
            uint c = lane;
            float acc = 0;
            for (uint k = 0; k < S; ++k) acc += Sc[sq*S + k] * Vt[k*DH + c];
            Ot[sq*DH + c] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Project Ot via W_o slice [h*DH..h*DH+DH, :] -> (S, D), ADD to y.
        // y[s, d] += sum_c Ot[s, c] * W_o[(h*DH + c)*D + d]
        // 32 sg, 2 rows each. Each lane handles 4 d outputs (d0 = lane*4).
        for (uint pass = 0; pass < 2; ++pass) {
            uint s = sg + pass * 32;
            uint d0 = lane * 4;
            float o0 = 0, o1 = 0, o2 = 0, o3 = 0;
            for (uint c = 0; c < DH; ++c) {
                float ov = Ot[s*DH + c];
                const uint wrow = (h*DH + c) * D + d0;
                o0 += ov * W_o[wrow + 0];
                o1 += ov * W_o[wrow + 1];
                o2 += ov * W_o[wrow + 2];
                o3 += ov * W_o[wrow + 3];
            }
            y[s*D + d0 + 0] += o0;
            y[s*D + d0 + 1] += o1;
            y[s*D + d0 + 2] += o2;
            y[s*D + d0 + 3] += o3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ============================================================
    // PHASE B: FFN
    // ============================================================
    // Compute LN2(y) and snapshot it into pool. Then FFN with contiguous reads.
    // pool[0..8191] = ln2 snapshot.
    threadgroup float* ln_snap = pool;
    {
        for (uint pass = 0; pass < 2; ++pass) {
            uint s = sg + pass * 32;
            uint off = s * D;
            uint d0 = lane * 4;
            float v0 = y[off + d0 + 0];
            float v1 = y[off + d0 + 1];
            float v2 = y[off + d0 + 2];
            float v3 = y[off + d0 + 3];
            float s1 = v0 + v1 + v2 + v3;
            float s2 = v0*v0 + v1*v1 + v2*v2 + v3*v3;
            s1 = simd_sum(s1);
            s2 = simd_sum(s2);
            float m = s1 / float(D);
            float var = max(s2/float(D) - m*m, 0.0f);
            float rs = rsqrt(var + eps);
            ln_snap[off + d0 + 0] = (v0 - m) * rs;
            ln_snap[off + d0 + 1] = (v1 - m) * rs;
            ln_snap[off + d0 + 2] = (v2 - m) * rs;
            ln_snap[off + d0 + 3] = (v3 - m) * rs;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // FFN: each sg handles 2 token rows (32 sg * 2 = 64 = S).
    // For row s:
    //   Step 1: hidden[FF=256] = LN @ W_ff1 (D x FF). gelu applied.
    //     Each lane owns 8 f-values: f = lane*8 + i, i=0..7.
    //     Strategy: for each d=0..127, load ln[s, d] (broadcast), load 8 contiguous W_ff1[d*FF + lane*8..+7], fmac.
    //     ln[s,d] needs broadcast — we put ln_snap in tg memory (already there), so just read.
    //   Step 2: out[D=128] = gelu(hidden) @ W_ff2 (FF x D). Add to y.
    //     Need hidden shared across sg's lanes. Lane owns 8 f-values, but to compute out[d] we need
    //     all f-values. Write hidden to a tg buffer per sg? 32 sg * 256 = 8192 floats — fits if we
    //     overwrite ln_snap. But ln_snap still needed? No, after Step 1 ln is consumed.
    //     Actually Step 1 needs ln for all d, Step 2 only needs hidden. So after Step 1 for a row,
    //     we can overwrite that row's ln_snap with hidden. But hidden is FF=256, ln_snap row is D=128.
    //     Layout mismatch.
    //
    //     Simpler: dedicate a per-sg hidden buffer in tg memory. 32 sg * FF=256 = 8192 floats.
    //     Reuse pool entirely AFTER ln_snap done. Each sg has its own slice pool[sg*256..sg*256+255].
    //
    //   So: compute hidden into pool[sg*FF..sg*FF+FF-1], then matmul out.
    //
    // But ln_snap and hidden need to coexist temporarily? No:
    //   - ln_snap occupies pool[0..S*D=8191]
    //   - we need ln for Step 1
    //   - After Step 1 for ALL rows finishes, we can overwrite pool with hidden... but Step 1 per row
    //     produces only that row's hidden. If sg handles 2 rows sequentially, between rows it produces
    //     hidden for row 1, then needs ln for row 2.
    //
    //   Solution: do both rows in parallel within sg? No, sg has 32 lanes and we use all for parallelism.
    //
    //   Per-row workflow: do Step 1 + Step 2 for row 1 entirely (hidden lives in registers across the f-loop),
    //   then move to row 2. But Step 2 needs full hidden vector — 256 floats — can't fit in registers
    //   (8 per lane = 256 total, distributed). We'd need cross-lane access during Step 2.
    //
    //   Cross-lane Step 2: hidden values are distributed (each lane owns 8 f's). For Step 2:
    //     out[d] = sum_f hidden[f] * W_ff2[f*D + d].
    //     Each lane owns d's (e.g. lane*4..lane*4+3). For each f, need broadcast(hidden[f]).
    //     hidden[f] is owned by lane = f/8, slot = f%8. Use simd_shuffle.
    //
    //   This avoids any cross-row tg traffic. Beautiful — purely register-resident.

    for (uint pass = 0; pass < 2; ++pass) {
        uint s = sg + pass * 32;
        uint roff = s * D;

        // Step 1: compute hidden[f=lane*8..+7], apply gelu.
        float h0=0, h1=0, h2=0, h3=0, h4=0, h5=0, h6=0, h7=0;
        const uint fbase = lane * 8;
        for (uint d = 0; d < D; ++d) {
            float ln_d = ln_snap[roff + d];
            uint wb = d * FF + fbase;
            float4 wa = *((device const float4*)(W_ff1 + wb));
            float4 wb2 = *((device const float4*)(W_ff1 + wb + 4));
            h0 += ln_d * wa.x;
            h1 += ln_d * wa.y;
            h2 += ln_d * wa.z;
            h3 += ln_d * wa.w;
            h4 += ln_d * wb2.x;
            h5 += ln_d * wb2.y;
            h6 += ln_d * wb2.z;
            h7 += ln_d * wb2.w;
        }
        const float k0 = 0.7978845608028654f;
        const float k1 = 0.044715f;
        #define GELU(x) (0.5f * (x) * (1.0f + precise::tanh(k0 * ((x) + k1*(x)*(x)*(x)))))
        h0 = GELU(h0); h1 = GELU(h1); h2 = GELU(h2); h3 = GELU(h3);
        h4 = GELU(h4); h5 = GELU(h5); h6 = GELU(h6); h7 = GELU(h7);
        #undef GELU

        // Step 2: out[d=lane*4..+3] = sum_f hidden[f] * W_ff2[f*D + d]
        // hidden[f] is owned by lane = f/8, slot f%8. Iterate f=0..255 broadcasting.
        const uint d0 = lane * 4;
        float o0=0, o1=0, o2=0, o3=0;
        // 32 source lanes, each owns 8 f's. Inner block of 8.
        for (uint src = 0; src < 32; ++src) {
            float g0 = simd_shuffle(h0, src);
            float g1 = simd_shuffle(h1, src);
            float g2 = simd_shuffle(h2, src);
            float g3 = simd_shuffle(h3, src);
            float g4 = simd_shuffle(h4, src);
            float g5 = simd_shuffle(h5, src);
            float g6 = simd_shuffle(h6, src);
            float g7 = simd_shuffle(h7, src);
            const float gs[8] = {g0,g1,g2,g3,g4,g5,g6,g7};
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8; ++i) {
                uint f = src * 8 + i;
                uint wb = f * D + d0;
                float gf = gs[i];
                float4 w = *((device const float4*)(W_ff2 + wb));
                o0 += gf * w.x;
                o1 += gf * w.y;
                o2 += gf * w.z;
                o3 += gf * w.w;
            }
        }
        y[roff + d0 + 0] += o0;
        y[roff + d0 + 1] += o1;
        y[roff + d0 + 2] += o2;
        y[roff + d0 + 3] += o3;
    }
}
