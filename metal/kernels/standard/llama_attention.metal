// llama_attention: GQA + RoPE attention (no KV cache, no LN).
#include <metal_stdlib>
using namespace metal;

#define S    64
#define D    128
#define HH   4
#define HKV  2
#define DH   32
#define G    2
#define TG   1024

// v3 optimizations vs baseline (24KB TG):
//   - Within a kv-group, fuse the G=2 query heads in the attention pass so each
//     Kh/Vh threadgroup read is reused across both heads (halves TG reads).
//   - float2 reads from Kh/Vh in the inner kt loop.
//   - Keep TG memory at 24KB (Kh + Vh + single oh) for occupancy parity.
//   - Process head 0 output projection while still holding head 1 attention
//     result in registers.
//
// TG memory: Kh (8KB) + Vh (8KB) + oh (8KB) = 24KB.

kernel void llama_attention_f32(
    device const float* x      [[buffer(0)]],
    device const float* W_qkv  [[buffer(1)]],
    device const float* W_o    [[buffer(2)]],
    device       float* y      [[buffer(3)]],
    constant uint& S_   [[buffer(4)]],
    constant uint& D_   [[buffer(5)]],
    constant uint& H_   [[buffer(6)]],
    constant uint& Hkv_ [[buffer(7)]],
    constant float& base [[buffer(8)]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    const uint t = tid.x;
    constexpr uint QKV_W = D + 2u * HKV * DH;   // 256

    threadgroup float Kh[S * DH];
    threadgroup float Vh[S * DH];
    threadgroup float oh[S * DH];

    const float inv_sqrt_dh = rsqrt(float(DH));
    const float two_over_dh = 2.0f / float(DH);
    const float log_base    = fast::log(base);

    const uint sq   = t >> 4;
    const uint lane = t & 15;
    const uint i_pair = lane;
    const uint dh0 = 2 * i_pair;
    const uint dh1 = dh0 + 1;
    const uint off = sq * D;

    const float pos    = float(sq);
    const float omega  = fast::exp(log_base * (-two_over_dh * float(i_pair)));
    const float ang    = pos * omega;
    const float c_rope = fast::cos(ang);
    const float s_rope = fast::sin(ang);

    // Fused projection of Q (all heads) and K, V (all kv-heads) in one sweep.
    float qreg0[HH], qreg1[HH];
    float kreg0[HKV], kreg1[HKV];
    float vreg0[HKV], vreg1[HKV];
    #pragma unroll
    for (uint h = 0; h < HH; ++h) { qreg0[h] = 0.f; qreg1[h] = 0.f; }
    #pragma unroll
    for (uint kv = 0; kv < HKV; ++kv) {
        kreg0[kv] = 0.f; kreg1[kv] = 0.f;
        vreg0[kv] = 0.f; vreg1[kv] = 0.f;
    }

    for (uint d = 0; d < D; ++d) {
        float xv = x[off + d];
        const device float2* wrow2 =
            (const device float2*)(W_qkv + d * QKV_W) + i_pair;
        #pragma unroll
        for (uint h = 0; h < HH; ++h) {
            float2 w = wrow2[h * (DH/2)];
            qreg0[h] += xv * w.x;
            qreg1[h] += xv * w.y;
        }
        const uint k_pair_off = D / 2;
        const uint v_pair_off = (D + HKV*DH) / 2;
        #pragma unroll
        for (uint kv = 0; kv < HKV; ++kv) {
            float2 wk = wrow2[k_pair_off + kv * (DH/2)];
            float2 wv = wrow2[v_pair_off + kv * (DH/2)];
            kreg0[kv] += xv * wk.x;
            kreg1[kv] += xv * wk.y;
            vreg0[kv] += xv * wv.x;
            vreg1[kv] += xv * wv.y;
        }
    }
    // RoPE Q.
    #pragma unroll
    for (uint h = 0; h < HH; ++h) {
        float q0 = qreg0[h], q1 = qreg1[h];
        qreg0[h] = q0 * c_rope - q1 * s_rope;
        qreg1[h] = q0 * s_rope + q1 * c_rope;
    }
    // RoPE K.
    #pragma unroll
    for (uint kv = 0; kv < HKV; ++kv) {
        float k0 = kreg0[kv], k1 = kreg1[kv];
        kreg0[kv] = k0 * c_rope - k1 * s_rope;
        kreg1[kv] = k0 * s_rope + k1 * c_rope;
    }

    for (uint kvh = 0; kvh < HKV; ++kvh) {
        Kh[sq*DH + dh0] = kreg0[kvh];
        Kh[sq*DH + dh1] = kreg1[kvh];
        Vh[sq*DH + dh0] = vreg0[kvh];
        Vh[sq*DH + dh1] = vreg1[kvh];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Q regs for both query heads in this kv-group.
        const float qa0 = qreg0[kvh*G + 0];
        const float qa1 = qreg1[kvh*G + 0];
        const float qb0 = qreg0[kvh*G + 1];
        const float qb1 = qreg1[kvh*G + 1];

        float outA0 = 0.f, outA1 = 0.f, mA = -INFINITY, lA = 0.f;
        float outB0 = 0.f, outB1 = 0.f, mB = -INFINITY, lB = 0.f;

        for (uint kt = 0; kt < S; ++kt) {
            const threadgroup float2* Kh2 = (const threadgroup float2*)(Kh + kt*DH);
            const threadgroup float2* Vh2 = (const threadgroup float2*)(Vh + kt*DH);
            float2 kv2 = Kh2[i_pair];
            float2 vv2 = Vh2[i_pair];

            // Head A.
            {
                float partial = qa0 * kv2.x + qa1 * kv2.y;
                partial += simd_shuffle_xor(partial, 1);
                partial += simd_shuffle_xor(partial, 2);
                partial += simd_shuffle_xor(partial, 4);
                partial += simd_shuffle_xor(partial, 8);
                float score = partial * inv_sqrt_dh;
                float mn = max(mA, score);
                float alpha = fast::exp(mA - mn);
                float beta  = fast::exp(score - mn);
                outA0 = outA0 * alpha + beta * vv2.x;
                outA1 = outA1 * alpha + beta * vv2.y;
                lA = lA * alpha + beta;
                mA = mn;
            }
            // Head B.
            {
                float partial = qb0 * kv2.x + qb1 * kv2.y;
                partial += simd_shuffle_xor(partial, 1);
                partial += simd_shuffle_xor(partial, 2);
                partial += simd_shuffle_xor(partial, 4);
                partial += simd_shuffle_xor(partial, 8);
                float score = partial * inv_sqrt_dh;
                float mn = max(mB, score);
                float alpha = fast::exp(mB - mn);
                float beta  = fast::exp(score - mn);
                outB0 = outB0 * alpha + beta * vv2.x;
                outB1 = outB1 * alpha + beta * vv2.y;
                lB = lB * alpha + beta;
                mB = mn;
            }
        }

        // Reuse Kh as a second oh buffer (Kh no longer needed for this kvh).
        // Barrier first: ensure all threads have finished reading Kh in the
        // attention loop before any thread overwrites it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* ohA = oh;
        threadgroup float* ohB = Kh;
        {
            float invA = 1.0f / lA;
            float invB = 1.0f / lB;
            ohA[sq*DH + dh0] = outA0 * invA;
            ohA[sq*DH + dh1] = outA1 * invA;
            ohB[sq*DH + dh0] = outB0 * invB;
            ohB[sq*DH + dh1] = outB1 * invB;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Fused output projection: y += ohA @ W_o[hA] + ohB @ W_o[hB].
        {
            const uint hA = kvh * G + 0;
            const uint hB = kvh * G + 1;
            for (uint idx = t; idx < S*D; idx += TG) {
                const uint sr   = idx / D;
                const uint dout = idx % D;
                float acc = 0.f;
                const threadgroup float* ohA_row = ohA + sr*DH;
                const threadgroup float* ohB_row = ohB + sr*DH;
                const device float* WoA = W_o + hA*DH*D + dout;
                const device float* WoB = W_o + hB*DH*D + dout;
                #pragma unroll 8
                for (uint dh = 0; dh < DH; ++dh) {
                    acc += ohA_row[dh] * WoA[dh*D];
                    acc += ohB_row[dh] * WoB[dh*D];
                }
                if (kvh == 0) { y[idx] = acc; } else { y[idx] += acc; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        }
    }
}
