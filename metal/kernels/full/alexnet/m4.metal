// alexnet-mini M4: lane-parallel output channels; minimal half conversions.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint A1H = 14, A1W = 14, A1C = 32;
constant constexpr uint A2H = 6,  A2W = 6,  A2C = 64;
constant constexpr uint A3H = 2,  A3W = 2,  A3C = 128;
constant constexpr uint FC1 = 256;

kernel void alexnet_f32(
    device const float* x      [[buffer(0)]],
    device const float* Wc1    [[buffer(1)]],
    device const float* Wc2    [[buffer(2)]],
    device const float* Wc3    [[buffer(3)]],
    device const float* Wfc1   [[buffer(4)]],
    device const float* Wfc2   [[buffer(5)]],
    device       float* y      [[buffer(6)]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  sgid      [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
    threadgroup half  a1[A1H * A1W * A1C];        // 12.5 KB
    threadgroup float a2[A2H * A2W * A2C];        //  9.0 KB
    threadgroup float a3[A3H * A3W * A3C];        //  2.0 KB
    threadgroup float fc1_buf[FC1];                //  1.0 KB
    threadgroup half Wc1c[32 * 75];                //  4.7 KB Wc1 cache (half precision)

    for (uint i = tid; i < 32u * 75u; i += 1024u) {
        Wc1c[i] = half(Wc1[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint spatial = sgid; spatial < A1H * A1W; spatial += 32u) {
        uint h2_ = spatial / A1W;
        uint w2  = spatial - h2_ * A1W;
        uint c = lane;
        threadgroup const half* Wf = Wc1c + c * 75u;
        uint hO0 = h2_ * 2u;
        uint wO0 = w2 * 2u;
        float pool0 = 0.0f, pool1 = 0.0f, pool2 = 0.0f, pool3 = 0.0f;
        #pragma unroll
        for (uint kh = 0; kh < 5u; ++kh) {
            #pragma unroll
            for (uint kw = 0; kw < 5u; ++kw) {
                uint wi = (kh * 5u + kw) * 3u;
                float w0 = float(Wf[wi+0]), w1 = float(Wf[wi+1]), w2v = float(Wf[wi+2]);
                uint xi00 = ((hO0 + 0 + kh) * 32u + (wO0 + 0 + kw)) * 3u;
                uint xi01 = ((hO0 + 0 + kh) * 32u + (wO0 + 1 + kw)) * 3u;
                uint xi10 = ((hO0 + 1 + kh) * 32u + (wO0 + 0 + kw)) * 3u;
                uint xi11 = ((hO0 + 1 + kh) * 32u + (wO0 + 1 + kw)) * 3u;
                pool0 = fma(x[xi00+0], w0, fma(x[xi00+1], w1, fma(x[xi00+2], w2v, pool0)));
                pool1 = fma(x[xi01+0], w0, fma(x[xi01+1], w1, fma(x[xi01+2], w2v, pool1)));
                pool2 = fma(x[xi10+0], w0, fma(x[xi10+1], w1, fma(x[xi10+2], w2v, pool2)));
                pool3 = fma(x[xi11+0], w0, fma(x[xi11+1], w1, fma(x[xi11+2], w2v, pool3)));
            }
        }
        float m = fmax(fmax(pool0, pool1), fmax(pool2, pool3));
        a1[(h2_ * A1W + w2) * A1C + c] = half(fmax(m, 0.0f));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint task = sgid; task < 72u; task += 32u) {
        uint c_half = task & 1u;
        uint spatial = task >> 1;
        uint h2_ = spatial / A2W;
        uint w2  = spatial - h2_ * A2W;
        uint c = c_half * 32u + lane;
        uint hO0 = h2_ * 2u;
        uint wO0 = w2 * 2u;
        float pool0 = 0.0f, pool1 = 0.0f, pool2 = 0.0f, pool3 = 0.0f;
        #pragma unroll
        for (uint kh = 0; kh < 3u; ++kh) {
            #pragma unroll
            for (uint kw = 0; kw < 3u; ++kw) {
                uint w_base = ((c * 3u + kh) * 3u + kw) * 32u;
                uint a_base00 = ((hO0 + 0 + kh) * A1W + (wO0 + 0 + kw)) * A1C;
                uint a_base01 = ((hO0 + 0 + kh) * A1W + (wO0 + 1 + kw)) * A1C;
                uint a_base10 = ((hO0 + 1 + kh) * A1W + (wO0 + 0 + kw)) * A1C;
                uint a_base11 = ((hO0 + 1 + kh) * A1W + (wO0 + 1 + kw)) * A1C;
                #pragma unroll
                for (uint ci = 0; ci < 32u; ++ci) {
                    float wv = Wc2[w_base + ci];
                    pool0 = fma(float(a1[a_base00 + ci]), wv, pool0);
                    pool1 = fma(float(a1[a_base01 + ci]), wv, pool1);
                    pool2 = fma(float(a1[a_base10 + ci]), wv, pool2);
                    pool3 = fma(float(a1[a_base11 + ci]), wv, pool3);
                }
            }
        }
        float m = fmax(fmax(pool0, pool1), fmax(pool2, pool3));
        a2[(h2_ * A2W + w2) * A2C + c] = fmax(m, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid < 16u) {
        uint c_quarter = sgid & 3u;
        uint spatial   = sgid >> 2;
        uint h2_ = spatial / A3W;
        uint w2  = spatial - h2_ * A3W;
        uint c = c_quarter * 32u + lane;
        uint hO0 = h2_ * 2u;
        uint wO0 = w2 * 2u;
        float pool0 = 0.0f, pool1 = 0.0f, pool2 = 0.0f, pool3 = 0.0f;
        #pragma unroll
        for (uint kh = 0; kh < 3u; ++kh) {
            #pragma unroll
            for (uint kw = 0; kw < 3u; ++kw) {
                uint w_base = ((c * 3u + kh) * 3u + kw) * 64u;
                uint a_base00 = ((hO0 + 0 + kh) * A2W + (wO0 + 0 + kw)) * A2C;
                uint a_base01 = ((hO0 + 0 + kh) * A2W + (wO0 + 1 + kw)) * A2C;
                uint a_base10 = ((hO0 + 1 + kh) * A2W + (wO0 + 0 + kw)) * A2C;
                uint a_base11 = ((hO0 + 1 + kh) * A2W + (wO0 + 1 + kw)) * A2C;
                #pragma unroll
                for (uint ci = 0; ci < 64u; ++ci) {
                    float wv = Wc3[w_base + ci];
                    pool0 = fma(a2[a_base00 + ci], wv, pool0);
                    pool1 = fma(a2[a_base01 + ci], wv, pool1);
                    pool2 = fma(a2[a_base10 + ci], wv, pool2);
                    pool3 = fma(a2[a_base11 + ci], wv, pool3);
                }
            }
        }
        float m = fmax(fmax(pool0, pool1), fmax(pool2, pool3));
        a3[(h2_ * A3W + w2) * A3C + c] = fmax(m, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float fc1_partial = 0.0f;
    uint fc1_o = sgid * 8u + (lane >> 2);
    uint fc1_k_part = lane & 3u;
    {
        uint k0 = fc1_k_part * 128u;
        #pragma unroll 8
        for (uint kk = 0; kk < 128u; ++kk) {
            uint k = k0 + kk;
            fc1_partial = fma(a3[k], Wfc1[k * 256u + fc1_o], fc1_partial);
        }
        fc1_partial += simd_shuffle_xor(fc1_partial, 1u);
        fc1_partial += simd_shuffle_xor(fc1_partial, 2u);
    }
    if (fc1_k_part == 0) {
        fc1_buf[fc1_o] = fmax(fc1_partial, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid < 10u) {
        uint o = sgid;
        float s = 0.0f;
        uint k0 = lane * 8u;
        #pragma unroll
        for (uint kk = 0; kk < 8u; ++kk) {
            s = fma(fc1_buf[k0 + kk], Wfc2[(k0 + kk) * 10u + o], s);
        }
        s = simd_sum(s);
        if (lane == 0) y[o] = s;
    }
}
