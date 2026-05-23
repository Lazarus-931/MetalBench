// densenet-mini: stem + 2 dense layers (channel concat) + GAP + FC.
// NHWC, padding=1, spatial=16x16, growth_rate=12, classes=10.
// Optimization: unrolled inner channel reduction (C0=12, Cin2=24 known constants).
#include <metal_stdlib>
using namespace metal;

kernel void densenet_f32(
    device const float* x       [[buffer(0)]],   // (1,16,16,3)
    device const float* W_stem  [[buffer(1)]],   // (12,3,3,3)
    device const float* W_d1    [[buffer(2)]],   // (12,3,3,12)
    device const float* W_d2    [[buffer(3)]],   // (12,3,3,24)
    device const float* W_fc    [[buffer(4)]],   // (36,10)
    device       float* y       [[buffer(5)]],   // (1,10)
    uint3 tid3                  [[thread_position_in_threadgroup]],
    uint  simd_lane             [[thread_index_in_simdgroup]],
    uint  simd_gid              [[simdgroup_index_in_threadgroup]])
{
    const uint tid = tid3.x;
    constexpr uint H = 16;
    constexpr uint W = 16;
    constexpr uint HW = H * W;
    constexpr uint C0 = 12;
    constexpr uint C1 = 12;
    constexpr uint Cin2 = 24;
    constexpr uint Cout2 = 12;
    constexpr uint Cfinal = 36;
    constexpr float inv_HW = 1.0f / float(HW);

    threadgroup float tg_h0[HW * C0];   // 12 KB
    threadgroup float tg_c1[HW * C1];   // 12 KB
    threadgroup float tg_Wd1[12u*3u*3u*12u];  // 5184 B
    threadgroup float gap[Cfinal];

    if (tid < Cfinal) gap[tid] = 0.0f;
    // Preload W_d1 into threadgroup memory (used heavily in Phase B).
    for (uint i = tid; i < 12u*3u*3u*12u; i += 256u) {
        tg_Wd1[i] = W_d1[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase A: stem conv (3 -> 12) + ReLU. Per-thread one (h,w), all 12 output channels.
    for (uint p = tid; p < HW; p += 256u) {
        uint h = p / W;
        uint w = p % W;
        float s[C0];
        #pragma clang loop unroll(full)
        for (uint c = 0; c < C0; ++c) s[c] = 0.0f;

        for (uint kh = 0; kh < 3u; ++kh) {
            int hi = int(h) + int(kh) - 1;
            if (hi < 0 || hi >= int(H)) continue;
            for (uint kw = 0; kw < 3u; ++kw) {
                int wi = int(w) + int(kw) - 1;
                if (wi < 0 || wi >= int(W)) continue;
                uint xb = (uint(hi) * W + uint(wi)) * 3u;
                float a0 = x[xb + 0], a1 = x[xb + 1], a2 = x[xb + 2];
                #pragma clang loop unroll(full)
                for (uint c = 0; c < C0; ++c) {
                    uint wb = ((c * 3u + kh) * 3u + kw) * 3u;
                    s[c] += a0 * W_stem[wb + 0]
                          + a1 * W_stem[wb + 1]
                          + a2 * W_stem[wb + 2];
                }
            }
        }
        #pragma clang loop unroll(full)
        for (uint c = 0; c < C0; ++c) {
            tg_h0[(h * W + w) * C0 + c] = fmax(s[c], 0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase B: d1 conv (12 -> 12) + ReLU. Reorder: each thread handles one (h,w) and
    // computes all 12 output channels at once, sharing activation loads.
    for (uint p = tid; p < HW; p += 256u) {
        uint h = p / W;
        uint w = p % W;
        float s[C1];
        #pragma clang loop unroll(full)
        for (uint c = 0; c < C1; ++c) s[c] = 0.0f;

        for (uint kh = 0; kh < 3u; ++kh) {
            int hi = int(h) + int(kh) - 1;
            if (hi < 0 || hi >= int(H)) continue;
            for (uint kw = 0; kw < 3u; ++kw) {
                int wi = int(w) + int(kw) - 1;
                if (wi < 0 || wi >= int(W)) continue;
                uint base_sp = (uint(hi) * W + uint(wi));
                threadgroup const float* sh0 = &tg_h0[base_sp * C0];
                float a[C0];
                #pragma clang loop unroll(full)
                for (uint ci = 0; ci < C0; ++ci) a[ci] = sh0[ci];
                #pragma clang loop unroll(full)
                for (uint c = 0; c < C1; ++c) {
                    uint wb = ((c * 3u + kh) * 3u + kw) * C0;
                    float acc = 0.0f;
                    #pragma clang loop unroll(full)
                    for (uint ci = 0; ci < C0; ++ci) {
                        acc += a[ci] * tg_Wd1[wb + ci];
                    }
                    s[c] += acc;
                }
            }
        }
        #pragma clang loop unroll(full)
        for (uint c = 0; c < C1; ++c) {
            tg_c1[(h * W + w) * C1 + c] = fmax(s[c], 0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase C: d2 conv (24 -> 12) + ReLU. Accumulate GAP[24..35] online while computing.
    // Parallelize over (h, w, c) producing exactly one c2 value per thread-iter; use atomic add via tree reduction over the 1024-thread group is overkill.
    // Instead: each thread accumulates a private partial for channel c, then reduce per channel.
    // Simpler: loop c in 0..Cout2; threads collaborate on HW positions.
    // Reordered: each thread loops over its HW positions once, accumulates 12 partials
    // (one per output channel). This visits each tg_h0/tg_c1 activation 9 times instead of 12*9.
    threadgroup float reduce_buf[8 * Cout2];  // 8 simdgroups × 12 channels = 96
    {
        float partial[Cout2];
        #pragma clang loop unroll(full)
        for (uint c = 0; c < Cout2; ++c) partial[c] = 0.0f;

        for (uint p = tid; p < HW; p += 256u) {
            uint h = p / W;
            uint w = p % W;
            float s[Cout2];
            #pragma clang loop unroll(full)
            for (uint c = 0; c < Cout2; ++c) s[c] = 0.0f;

            for (uint kh = 0; kh < 3u; ++kh) {
                int hi = int(h) + int(kh) - 1;
                if (hi < 0 || hi >= int(H)) continue;
                for (uint kw = 0; kw < 3u; ++kw) {
                    int wi = int(w) + int(kw) - 1;
                    if (wi < 0 || wi >= int(W)) continue;
                    uint base_sp = (uint(hi) * W + uint(wi));
                    threadgroup const float* sh0 = &tg_h0[base_sp * C0];
                    threadgroup const float* sc1 = &tg_c1[base_sp * C1];
                    // Load 24-vector activation once.
                    float a[Cin2];
                    #pragma clang loop unroll(full)
                    for (uint ci = 0; ci < C0; ++ci) a[ci] = sh0[ci];
                    #pragma clang loop unroll(full)
                    for (uint ci = 0; ci < C1; ++ci) a[C0 + ci] = sc1[ci];
                    // For each output channel, accumulate 24 FMAs.
                    #pragma clang loop unroll(full)
                    for (uint c = 0; c < Cout2; ++c) {
                        uint wb = ((c * 3u + kh) * 3u + kw) * Cin2;
                        float acc = 0.0f;
                        #pragma clang loop unroll(full)
                        for (uint ci = 0; ci < Cin2; ++ci) {
                            acc += a[ci] * W_d2[wb + ci];
                        }
                        s[c] += acc;
                    }
                }
            }
            #pragma clang loop unroll(full)
            for (uint c = 0; c < Cout2; ++c) partial[c] += fmax(s[c], 0.0f);
        }

        // Reduce: 8 simdgroups × 32 lanes. Per-channel simd_sum then cross-simdgroup.
        #pragma clang loop unroll(full)
        for (uint c = 0; c < Cout2; ++c) {
            float lane_sum = simd_sum(partial[c]);
            if (simd_lane == 0) reduce_buf[simd_gid * Cout2 + c] = lane_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_gid == 0 && simd_lane < Cout2) {
            uint c = simd_lane;
            float tot = 0.0f;
            #pragma clang loop unroll(full)
            for (uint g = 0; g < 8u; ++g) tot += reduce_buf[g * Cout2 + c];
            gap[Cin2 + c] = tot * inv_HW;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // GAP for channels 0..23 (h0 and c1). Each simdgroup handles channels via simd_sum.
    // 8 simdgroups × 32 lanes = 256 threads; assign simdgroup to channel index.
    // We have 24 channels, 8 simdgroups → each does 3 channels.
    {
        for (uint k = 0; k < 3u; ++k) {
            uint cc = simd_gid * 3u + k;
            if (cc < C0 + C1) {
                // 32 lanes split HW=256 → 8 spatial each
                float ps = 0.0f;
                #pragma clang loop unroll(full)
                for (uint q = 0; q < 8u; ++q) {
                    uint p = simd_lane * 8u + q;
                    if (cc < C0) ps += tg_h0[p * C0 + cc];
                    else         ps += tg_c1[p * C1 + (cc - C0)];
                }
                float tot = simd_sum(ps);
                if (simd_lane == 0) gap[cc] = tot * inv_HW;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // FC.
    if (tid < 10u) {
        float s = 0.0f;
        for (uint k = 0; k < Cfinal; ++k) s += gap[k] * W_fc[k * 10u + tid];
        y[tid] = s;
    }
}
