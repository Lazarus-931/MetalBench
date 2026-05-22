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

// Optimizations vs baseline:
//   - Project Q for ALL HH heads in one fused sweep of x and store in regs.
//   - fast::exp / fast::cos / fast::sin / fast::log throughout.
//   - RoPE coefficients computed once per thread (shared by Q and K).
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

    // ---- Fused projection of Q (all heads) and K, V (all kv-heads) in ONE
    //      sweep of x. ----
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

    // Use float2 loads for the column pair (dh0, dh1 = 2*lane, 2*lane+1).
    for (uint d = 0; d < D; ++d) {
        float xv = x[off + d];
        const device float2* wrow2 =
            (const device float2*)(W_qkv + d * QKV_W) + i_pair;  // each elt is 2 consecutive cols
        // Q heads occupy columns [0, HH*DH) = 16 pairs per head row, so
        // for head h the pair index is h*16 + i_pair = h*(DH/2) + i_pair.
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
    // Apply RoPE to Q (in regs).
    #pragma unroll
    for (uint h = 0; h < HH; ++h) {
        float q0 = qreg0[h], q1 = qreg1[h];
        qreg0[h] = q0 * c_rope - q1 * s_rope;
        qreg1[h] = q0 * s_rope + q1 * c_rope;
    }
    // Apply RoPE to K (in regs).
    #pragma unroll
    for (uint kv = 0; kv < HKV; ++kv) {
        float k0 = kreg0[kv], k1 = kreg1[kv];
        kreg0[kv] = k0 * c_rope - k1 * s_rope;
        kreg1[kv] = k0 * s_rope + k1 * c_rope;
    }

    // ---- Loop over kv-heads. ----
    for (uint kvh = 0; kvh < HKV; ++kvh) {
        // Pull this kvh's K, V from registers into TG memory.
        Kh[sq*DH + dh0] = kreg0[kvh];
        Kh[sq*DH + dh1] = kreg1[kvh];
        Vh[sq*DH + dh0] = vreg0[kvh];
        Vh[sq*DH + dh1] = vreg1[kvh];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint h_off = 0; h_off < G; ++h_off) {
            const uint h  = kvh * G + h_off;
            const float q0r = qreg0[h];
            const float q1r = qreg1[h];

            float out0 = 0.f, out1 = 0.f;
            float m_run = -INFINITY;
            float l_run = 0.f;
            for (uint kt = 0; kt < S; ++kt) {
                float ka = Kh[kt*DH + dh0];
                float kb = Kh[kt*DH + dh1];
                float partial = q0r * ka + q1r * kb;
                partial += simd_shuffle_xor(partial, 1);
                partial += simd_shuffle_xor(partial, 2);
                partial += simd_shuffle_xor(partial, 4);
                partial += simd_shuffle_xor(partial, 8);
                float score = partial * inv_sqrt_dh;
                float mn = max(m_run, score);
                float alpha = fast::exp(m_run - mn);
                float beta  = fast::exp(score - mn);
                out0 = out0 * alpha + beta * Vh[kt*DH + dh0];
                out1 = out1 * alpha + beta * Vh[kt*DH + dh1];
                l_run = l_run * alpha + beta;
                m_run = mn;
            }
            float inv = 1.0f / l_run;
            oh[sq*DH + dh0] = out0 * inv;
            oh[sq*DH + dh1] = out1 * inv;

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint idx = t; idx < S*D; idx += TG) {
                const uint sr   = idx / D;
                const uint dout = idx % D;
                float acc = 0.f;
                for (uint dh = 0; dh < DH; ++dh) {
                    acc += oh[sr*DH + dh] * W_o[(h*DH + dh)*D + dout];
                }
                if (kvh == 0 && h_off == 0) {
                    y[idx] = acc;
                } else {
                    y[idx] += acc;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        }
    }
}
