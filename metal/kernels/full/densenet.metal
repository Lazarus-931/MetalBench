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
    threadgroup float tg_Wd2[12u*3u*3u*24u];  // 10368 B
    threadgroup float gap[Cfinal];

    if (tid < Cfinal) gap[tid] = 0.0f;
    // Preload W_d1 into threadgroup memory (used heavily in Phase B).
    for (uint i = tid; i < 12u*3u*3u*12u; i += 256u) {
        tg_Wd1[i] = W_d1[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase A: stem conv (3 -> 12) + ReLU
    for (uint i = tid; i < HW * C0; i += 256u) {
        uint h = i / (W * C0);
        uint t = i % (W * C0);
        uint w = t / C0;
        uint c = t % C0;
        float s = 0.0f;
        for (uint kh = 0; kh < 3u; ++kh) {
            int hi = int(h) + int(kh) - 1;
            if (hi < 0 || hi >= int(H)) continue;
            for (uint kw = 0; kw < 3u; ++kw) {
                int wi = int(w) + int(kw) - 1;
                if (wi < 0 || wi >= int(W)) continue;
                uint xb = (uint(hi) * W + uint(wi)) * 3u;
                uint wb = ((c * 3u + kh) * 3u + kw) * 3u;
                s += x[xb + 0] * W_stem[wb + 0]
                   + x[xb + 1] * W_stem[wb + 1]
                   + x[xb + 2] * W_stem[wb + 2];
            }
        }
        tg_h0[(h * W + w) * C0 + c] = fmax(s, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase B: d1 conv (12 -> 12) + ReLU; inner ci loop fully unrolled (C0=12).
    for (uint i = tid; i < HW * C1; i += 256u) {
        uint h = i / (W * C1);
        uint t = i % (W * C1);
        uint w = t / C1;
        uint c = t % C1;
        float s = 0.0f;
        for (uint kh = 0; kh < 3u; ++kh) {
            int hi = int(h) + int(kh) - 1;
            if (hi < 0 || hi >= int(H)) continue;
            for (uint kw = 0; kw < 3u; ++kw) {
                int wi = int(w) + int(kw) - 1;
                if (wi < 0 || wi >= int(W)) continue;
                uint sb = (uint(hi) * W + uint(wi)) * C0;
                uint wb = ((c * 3u + kh) * 3u + kw) * C0;
                #pragma clang loop unroll(full)
                for (uint ci = 0; ci < C0; ++ci) {
                    s += tg_h0[sb + ci] * tg_Wd1[wb + ci];
                }
            }
        }
        tg_c1[(h * W + w) * C1 + c] = fmax(s, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase C: d2 conv (24 -> 12) + ReLU. Accumulate GAP[24..35] online while computing.
    // Parallelize over (h, w, c) producing exactly one c2 value per thread-iter; use atomic add via tree reduction over the 1024-thread group is overkill.
    // Instead: each thread accumulates a private partial for channel c, then reduce per channel.
    // Simpler: loop c in 0..Cout2; threads collaborate on HW positions.
    threadgroup float reduce_buf[8];  // 8 simdgroups in 256-thread tg
    for (uint c = 0; c < Cout2; ++c) {
        float partial = 0.0f;
        for (uint p = tid; p < HW; p += 256u) {
            uint h = p / W;
            uint w = p % W;
            float s = 0.0f;
            for (uint kh = 0; kh < 3u; ++kh) {
                int hi = int(h) + int(kh) - 1;
                if (hi < 0 || hi >= int(H)) continue;
                for (uint kw = 0; kw < 3u; ++kw) {
                    int wi = int(w) + int(kw) - 1;
                    if (wi < 0 || wi >= int(W)) continue;
                    uint base_sp = (uint(hi) * W + uint(wi));
                    uint wb = ((c * 3u + kh) * 3u + kw) * Cin2;
                    threadgroup const float* sh0 = &tg_h0[base_sp * C0];
                    threadgroup const float* sc1 = &tg_c1[base_sp * C1];
                    #pragma clang loop unroll(full)
                    for (uint ci = 0; ci < C0; ++ci) {
                        s += sh0[ci] * W_d2[wb + ci];
                    }
                    #pragma clang loop unroll(full)
                    for (uint ci = 0; ci < C1; ++ci) {
                        s += sc1[ci] * W_d2[wb + C0 + ci];
                    }
                }
            }
            partial += fmax(s, 0.0f);
        }
        float lane_sum = simd_sum(partial);
        if (simd_lane == 0) reduce_buf[simd_gid] = lane_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_gid == 0) {
            float v = (simd_lane < 8u) ? reduce_buf[simd_lane] : 0.0f;
            float total = simd_sum(v);
            if (simd_lane == 0) gap[Cin2 + c] = total * inv_HW;
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
