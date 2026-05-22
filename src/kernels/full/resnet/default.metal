// resnet-mini: stem + 1 residual block + GAP + FC. NHWC, padding=1.
#include <metal_stdlib>
using namespace metal;

kernel void resnet_f32(
    device const float* x       [[buffer(0)]],
    device const float* W_stem  [[buffer(1)]],
    device const float* W_a     [[buffer(2)]],
    device const float* W_b     [[buffer(3)]],
    device const float* W_fc    [[buffer(4)]],
    device       float* y       [[buffer(5)]],
    uint3 tid3                  [[thread_position_in_threadgroup]])
{
    const uint tid = tid3.x;

    threadgroup float h_stem[32 * 32 * 16];  // 64 KB — exceeds TG limit

    threadgroup float tg_block_out[32 * 32 * 16];

    for (uint i = tid; i < 32u * 32u * 16u; i += 1024u) {
        uint h = i / (32u * 16u);
        uint t = i % (32u * 16u);
        uint w = t / 16u;
        uint c = t % 16u;
        float s = 0.0f;
        for (uint kh = 0; kh < 3u; ++kh)
            for (uint kw = 0; kw < 3u; ++kw)
                for (uint ci = 0; ci < 3u; ++ci) {
                    int hi = int(h) + int(kh) - 1;
                    int wi = int(w) + int(kw) - 1;
                    if (hi < 0 || hi >= 32 || wi < 0 || wi >= 32) continue;
                    s += x[(uint(hi) * 32u + uint(wi)) * 3u + ci]
                       * W_stem[((c * 3u + kh) * 3u + kw) * 3u + ci];
                }
        h_stem[(h * 32u + w) * 16u + c] = fmax(s, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < 32u * 32u * 16u; i += 1024u) {
        uint h = i / (32u * 16u);
        uint t = i % (32u * 16u);
        uint w = t / 16u;
        uint c = t % 16u;
        float s = 0.0f;
        for (uint kh = 0; kh < 3u; ++kh)
            for (uint kw = 0; kw < 3u; ++kw)
                for (uint ci = 0; ci < 16u; ++ci) {
                    int hi = int(h) + int(kh) - 1;
                    int wi = int(w) + int(kw) - 1;
                    if (hi < 0 || hi >= 32 || wi < 0 || wi >= 32) continue;
                    s += h_stem[(uint(hi) * 32u + uint(wi)) * 16u + ci]
                       * W_a[((c * 3u + kh) * 3u + kw) * 16u + ci];
                }
        tg_block_out[(h * 32u + w) * 16u + c] = fmax(s, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < 32u * 32u * 16u; i += 1024u) {
        uint h = i / (32u * 16u);
        uint t = i % (32u * 16u);
        uint w = t / 16u;
        uint c = t % 16u;
        float s = 0.0f;
        for (uint kh = 0; kh < 3u; ++kh)
            for (uint kw = 0; kw < 3u; ++kw)
                for (uint ci = 0; ci < 16u; ++ci) {
                    int hi = int(h) + int(kh) - 1;
                    int wi = int(w) + int(kw) - 1;
                    if (hi < 0 || hi >= 32 || wi < 0 || wi >= 32) continue;
                    s += tg_block_out[(uint(hi) * 32u + uint(wi)) * 16u + ci]
                       * W_b[((c * 3u + kh) * 3u + kw) * 16u + ci];
                }
        h_stem[(h * 32u + w) * 16u + c] = fmax(s + h_stem[(h * 32u + w) * 16u + c], 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float gap[16];
    if (tid < 16) {
        float s = 0.0f;
        for (uint h = 0; h < 32u; ++h)
            for (uint w = 0; w < 32u; ++w)
                s += h_stem[(h * 32u + w) * 16u + tid];
        gap[tid] = s / float(32u * 32u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 10) {
        float s = 0.0f;
        for (uint k = 0; k < 16u; ++k) s += gap[k] * W_fc[k * 10u + tid];
        y[tid] = s;
    }
}
