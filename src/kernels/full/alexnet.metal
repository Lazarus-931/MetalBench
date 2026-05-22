// alexnet-mini: 3 conv (k=5/3/3) + 3 maxpool 2x2 + 2 fc. Single TG, naive sequential.
// All activation buffers live in device output buffer (recycled). Total work per
// dispatch dominated by conv1 (32*5*5*3 = 2400 weights, 28*28*32 outputs).
#include <metal_stdlib>
using namespace metal;

// Hard-coded shapes:
// x  (1, 32, 32, 3); Wc1 (32, 5, 5, 3); Wc2 (64, 3, 3, 32); Wc3 (128, 3, 3, 64);
// Wfc1 (512, 256); Wfc2 (256, 10)
// Pool stages: 28→14, 12→6, 4→2. Flatten: 2*2*128 = 512.

kernel void alexnet_f32(
    device const float* x      [[buffer(0)]],
    device const float* Wc1    [[buffer(1)]],
    device const float* Wc2    [[buffer(2)]],
    device const float* Wc3    [[buffer(3)]],
    device const float* Wfc1   [[buffer(4)]],
    device const float* Wfc2   [[buffer(5)]],
    device       float* y      [[buffer(6)]],
    uint3 tid3                 [[thread_position_in_threadgroup]])
{
    const uint tid = tid3.x;
    threadgroup float a1[28 * 28 * 32];   // 100352 floats — too big. shrink layout below.
    // Realistically use device scratch — but harness only routes one output. We use
    // y itself as scratch since it's much bigger than the (1, 10) output we need.
    // Output (1, 10) → only first 10 floats matter at end. Use rest as scratch.

    // For simplicity, this kernel writes intermediate activations into TG memory
    // a1 (after conv1 + maxpool); a2 (after conv2 + maxpool); a3 (after conv3 + maxpool).
    // a1: 14*14*32 = 6272
    // a2: 6*6*64 = 2304
    // a3: 2*2*128 = 512
    // total tg mem: ~9088 floats = ~35 KB — fits.
    threadgroup float tg_a1[14 * 14 * 32];
    threadgroup float tg_a2[6 * 6 * 64];
    threadgroup float tg_a3[2 * 2 * 128];
    threadgroup float tg_fc1[256];

    // ---- conv1: x (1,32,32,3) @ Wc1 (32,5,5,3) → (1, 28, 28, 32), then maxpool 2x2 → (1,14,14,32) ----
    for (uint i = tid; i < 14u * 14u * 32u; i += 1024u) {
        uint h2 = i / (14u * 32u);
        uint t  = i % (14u * 32u);
        uint w2 = t / 32u;
        uint c  = t % 32u;
        float m = -INFINITY;
        for (uint ph = 0; ph < 2u; ++ph) for (uint pw = 0; pw < 2u; ++pw) {
            uint hO = h2 * 2u + ph;   // 0..27
            uint wO = w2 * 2u + pw;
            float s = 0.0f;
            for (uint kh = 0; kh < 5u; ++kh)
                for (uint kw = 0; kw < 5u; ++kw)
                    for (uint ci = 0; ci < 3u; ++ci)
                        s += x[((hO + kh) * 32u + (wO + kw)) * 3u + ci]
                           * Wc1[((c * 5u + kh) * 5u + kw) * 3u + ci];
            m = fmax(m, s);
        }
        tg_a1[(h2 * 14u + w2) * 32u + c] = fmax(m, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- conv2: a1 (14,14,32) @ Wc2 (64,3,3,32) → (12,12,64), maxpool → (6,6,64) ----
    for (uint i = tid; i < 6u * 6u * 64u; i += 1024u) {
        uint h2 = i / (6u * 64u);
        uint t  = i % (6u * 64u);
        uint w2 = t / 64u;
        uint c  = t % 64u;
        float m = -INFINITY;
        for (uint ph = 0; ph < 2u; ++ph) for (uint pw = 0; pw < 2u; ++pw) {
            uint hO = h2 * 2u + ph;
            uint wO = w2 * 2u + pw;
            float s = 0.0f;
            for (uint kh = 0; kh < 3u; ++kh)
                for (uint kw = 0; kw < 3u; ++kw)
                    for (uint ci = 0; ci < 32u; ++ci)
                        s += tg_a1[((hO + kh) * 14u + (wO + kw)) * 32u + ci]
                           * Wc2[((c * 3u + kh) * 3u + kw) * 32u + ci];
            m = fmax(m, s);
        }
        tg_a2[(h2 * 6u + w2) * 64u + c] = fmax(m, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- conv3: a2 (6,6,64) @ Wc3 (128,3,3,64) → (4,4,128), maxpool → (2,2,128) ----
    for (uint i = tid; i < 2u * 2u * 128u; i += 1024u) {
        uint h2 = i / (2u * 128u);
        uint t  = i % (2u * 128u);
        uint w2 = t / 128u;
        uint c  = t % 128u;
        float m = -INFINITY;
        for (uint ph = 0; ph < 2u; ++ph) for (uint pw = 0; pw < 2u; ++pw) {
            uint hO = h2 * 2u + ph;
            uint wO = w2 * 2u + pw;
            float s = 0.0f;
            for (uint kh = 0; kh < 3u; ++kh)
                for (uint kw = 0; kw < 3u; ++kw)
                    for (uint ci = 0; ci < 64u; ++ci)
                        s += tg_a2[((hO + kh) * 6u + (wO + kw)) * 64u + ci]
                           * Wc3[((c * 3u + kh) * 3u + kw) * 64u + ci];
            m = fmax(m, s);
        }
        tg_a3[(h2 * 2u + w2) * 128u + c] = fmax(m, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- fc1: a3 flatten (512) @ Wfc1 (512, 256) → 256, ReLU ----
    for (uint i = tid; i < 256u; i += 1024u) {
        float s = 0.0f;
        for (uint k = 0; k < 512u; ++k) s += tg_a3[k] * Wfc1[k * 256u + i];
        tg_fc1[i] = fmax(s, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- fc2: fc1 (256) @ Wfc2 (256, 10) → 10 ----
    for (uint i = tid; i < 10u; i += 1024u) {
        float s = 0.0f;
        for (uint k = 0; k < 256u; ++k) s += tg_fc1[k] * Wfc2[k * 10u + i];
        y[i] = s;
    }
}
