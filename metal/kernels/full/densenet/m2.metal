// densenet-mini default variant (M2-tuned): Phase B + Phase C use the per-thread
// "compute all C output channels at once" reorder; Phase A keeps the simple
// per-channel mapping which on M2 was faster than the reorder.
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

    threadgroup float tg_h0[HW * C0];
    threadgroup float tg_c1[HW * C1];
    threadgroup float tg_Wd1[12u*3u*3u*12u];
    threadgroup float gap[Cfinal];

    if (tid < Cfinal) gap[tid] = 0.0f;
    for (uint i = tid; i < 12u*3u*3u*12u; i += 256u) {
        tg_Wd1[i] = W_d1[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase A: stem conv (3 -> 12) + ReLU — simple per-channel mapping.
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

    // Phase B reorder: each thread one (h,w), all 12 output channels at once.
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

    // Phase C reorder: per-thread spatial, 12 output channel accumulators.
    threadgroup float reduce_buf[8 * Cout2];
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
                    float a[Cin2];
                    #pragma clang loop unroll(full)
                    for (uint ci = 0; ci < C0; ++ci) a[ci] = sh0[ci];
                    #pragma clang loop unroll(full)
                    for (uint ci = 0; ci < C1; ++ci) a[C0 + ci] = sc1[ci];
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

    // GAP for channels 0..23.
    {
        for (uint k = 0; k < 3u; ++k) {
            uint cc = simd_gid * 3u + k;
            if (cc < C0 + C1) {
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

    if (tid < 10u) {
        float s = 0.0f;
        for (uint k = 0; k < Cfinal; ++k) s += gap[k] * W_fc[k * 10u + tid];
        y[tid] = s;
    }
}
