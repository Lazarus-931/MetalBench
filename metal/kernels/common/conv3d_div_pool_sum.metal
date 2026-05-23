// conv3d -> div -> avg_pool2 -> sum. Single-threadgroup, 1024 threads = 32 simdgroups.
#include <metal_stdlib>
using namespace metal;

kernel void conv3d_div_pool_sum_f32(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device       float* y [[buffer(2)]],
    constant     uint& N  [[buffer(3)]],
    constant     uint& C  [[buffer(4)]],
    constant     uint& D  [[buffer(5)]],
    constant     uint& H  [[buffer(6)]],
    constant     uint& W  [[buffer(7)]],
    constant     uint& K  [[buffer(8)]],
    constant     uint& R  [[buffer(9)]],
    constant     float& div_val  [[buffer(10)]],
    constant     float& bias_val [[buffer(11)]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg  [[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]])
{
    constexpr uint Cc=32, Rc=3, Dc=32, Hc=32, Wc=32, Nc=4;
    constexpr uint D2=Dc-Rc+1, H2=Hc-Rc+1, W2=Wc-Rc+1;
    constexpr uint VOL = D2*H2*W2;
    constexpr uint PATCH = Cc*Rc*Rc*Rc;      // 864
    constexpr uint TG = 1024;
    constexpr uint NSG = TG / 32;            // 32

    constexpr uint TPB = 10;  // tiles per batch between barriers
    threadgroup float sgmax_buf[2][TPB][NSG];

    const float final_scale = (1.0f / div_val) / float(VOL);

    // Preload weight slices for k0=sg, k1=sg+32 into registers.
    float wreg0[27];
    float wreg1[27];
    {
        const device float* wk0 = w + (uint)sg          * PATCH;
        const device float* wk1 = w + (uint)(sg + 32u)  * PATCH;
        #pragma unroll
        for (uint j = 0; j < 27; ++j) {
            uint off = lane + 32u*j;
            wreg0[j] = wk0[off];
            wreg1[j] = wk1[off];
        }
    }

    // Precompute per-lane (rd,rh,rw,cvec_off) for each j=0..26.
    // i = lane + 32*j. spatial = i/8 (0..107). cvec = i & 7 (which float4 within C=32).
    // spatial = rd*9 + rh*3 + rw. Each spatial position has 8 vec4s (32 floats / 4).
    // But here we want a single float per lane (not vec4): the patch[i] is a SCALAR.
    // patch was loaded as p4[i] = x4[x_idx + cvec], so patch float i corresponds to
    // x[n,d2+rd,h2+rh,w2+rw, (cvec*4 + (i mod 4))]. Let me recompute carefully:
    //   p4 indexed by i (vec4-index 0..215). For each i: spatial=i>>3, cvec=i&7.
    //   p4[i] is a float4 of the 4 channel-elements (cvec*4 .. cvec*4+3) at that spatial.
    // But we read patch[lane + 32*j] which is a SCALAR (float). lane+32*j is a float index 0..863.
    // patch float index I corresponds to p4 vec4 index I>>2, lane within vec4 I&3.
    // I = lane + 32*j. I>>2 = (lane>>2) + 8*j  (since 32/4=8). I&3 = lane & 3.
    // So float at I lives in p4 vec4 index P = (lane>>2) + 8*j, scalar offset (lane & 3).
    // p4[P] = x4[x_idx_base + cvec], where for P: spatial = P>>3 = ((lane>>2) + 8*j) >> 3.
    //   For j in [0,27), P ranges (lane>>2) + 0,8,16,...,208. spatial = P>>3, cvec = P&7.
    //   With lane>>2 in [0,8): P = (lane>>2) + 8j. P&7 = lane>>2. P>>3 = j.
    //   So spatial = j, cvec = lane>>2. spatial = j means rd = j/9, rh = (j%9)/3, rw = j%3.
    // Channel index within C=32: c = cvec*4 + (lane & 3) = (lane>>2)*4 + (lane & 3)
    //                          = (lane & ~3) + (lane & 3) = lane.  (since lane<32)
    // So patch[lane + 32*j] = x[n, d2 + rd(j), h2 + rh(j), w2 + rw(j), lane].
    //
    // BEAUTIFUL: each lane reads channel c=lane for j=0..26 at (rd,rh,rw)=(j/9,(j%9)/3,j%3).

    float running_sum = 0.0f;
    uint pp = 0;

    // Precompute the 27 (rd,rh,rw) offsets as flat x-index strides at lane channel.
    // Stride per (rd,rh,rw): rd*Hc*Wc*Cc + rh*Wc*Cc + rw*Cc.
    // We'll compute base = ((n*Dc+d2)*Hc+h2)*Wc+w2)*Cc + lane, then add stride[j].
    uint strides[27];
    #pragma unroll
    for (uint j = 0; j < 27; ++j) {
        uint rd = j / 9u;
        uint rem = j - rd*9u;
        uint rh = rem / 3u;
        uint rw = rem - rh*3u;
        strides[j] = (rd*Hc + rh)*Wc*Cc + rw*Cc;
    }

    for (uint n = 0; n < Nc; ++n) {
        for (uint d2 = 0; d2 < D2; ++d2) {
        for (uint h2 = 0; h2 < H2; ++h2) {
        // W2=30, batches of TPB=3 → 10 batches.
        for (uint w2b = 0; w2b < W2; w2b += TPB) {
            float per_sg_max_t[TPB];
            #pragma unroll
            for (uint t = 0; t < TPB; ++t) {
                uint w2 = w2b + t;
                uint base = ((((n*Dc + d2)*Hc + h2)*Wc + w2)*Cc) + lane;
                float preg[27];
                #pragma unroll
                for (uint j = 0; j < 27; ++j) preg[j] = x[base + strides[j]];
                float s0 = 0.0f, s1 = 0.0f;
                #pragma unroll
                for (uint j = 0; j < 27; ++j) {
                    s0 = fma(preg[j], wreg0[j], s0);
                    s1 = fma(preg[j], wreg1[j], s1);
                }
                s0 = simd_sum(s0);
                s1 = simd_sum(s1);
                per_sg_max_t[t] = max(s0, s1);
            }

            if (lane == 0) {
                #pragma unroll
                for (uint t = 0; t < TPB; ++t)
                    sgmax_buf[pp][t][sg] = per_sg_max_t[t];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float acc = 0.0f;
            #pragma unroll
            for (uint t = 0; t < TPB; ++t) {
                float v = sgmax_buf[pp][t][lane];
                v = simd_max(v);
                acc += v;
            }
            running_sum += acc;
            pp ^= 1u;
        }}}
    }

    if (tid == 0) {
        y[0] = running_sum * final_scale + float(Nc) * bias_val;
    }
}
