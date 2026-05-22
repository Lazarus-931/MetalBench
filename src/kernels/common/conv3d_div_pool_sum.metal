// Single-threadgroup design (1024 threads = 32 simdgroups).
// One TG iterates over all (n,d2,h2,w2) tiles. Per tile:
//   - Cooperatively load patch (864 f32) into tg memory.
//   - 32 simdgroups each compute dot products for 2 k's (k=sg and k=sg+32),
//     32 lanes splitting the 864 elements (27 per lane), then simd_sum.
//   - max over (s0, s1) within sg → per_sg_max
//   - cross-simdgroup max reduce using tg mem + one simdgroup
// All threads keep running_sum; finally thread 0 writes y[0].
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
    constexpr uint PV4 = PATCH / 4;          // 216

    threadgroup float patch[PATCH];
    threadgroup float sgmax[NSG];
    threadgroup float bcast;                  // broadcast slot for tile_max

    const float final_scale = (1.0f / div_val) / float(VOL);

    // Preload weight slices for k0=sg, k1=sg+32 into registers (27 each, strided).
    float wreg0[27];
    float wreg1[27];
    {
        const device float* wk0 = w + (uint)sg          * PATCH;
        const device float* wk1 = w + (uint)(sg + 32u)  * PATCH;
        for (uint j = 0; j < 27; ++j) {
            wreg0[j] = wk0[lane + 32u*j];
            wreg1[j] = wk1[lane + 32u*j];
        }
    }

    device const float4* x4 = (device const float4*)x;
    threadgroup float4* p4  = (threadgroup float4*)patch;

    float running_sum = 0.0f;

    for (uint n = 0; n < Nc; ++n) {
        for (uint d2 = 0; d2 < D2; ++d2) {
        for (uint h2 = 0; h2 < H2; ++h2) {
        for (uint w2 = 0; w2 < W2; ++w2) {
            // Load patch: 216 vec4s, 1024 threads.
            // Strided: thread tid fetches vec4 idx tid mod 216 if tid<216, else skip.
            if (tid < PV4) {
                uint i = tid;
                uint spatial = i >> 3;
                uint cvec    = i & 7;
                uint rd = spatial / 9u;
                uint rem= spatial - rd*9u;
                uint rh = rem / 3u;
                uint rw = rem - rh*3u;
                uint x_idx = ((((n*Dc + d2 + rd)*Hc + h2 + rh)*Wc + w2 + rw)*Cc) >> 2;
                p4[i] = x4[x_idx + cvec];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float s0 = 0.0f, s1 = 0.0f;
            // 27 fmas
            #pragma unroll
            for (uint j = 0; j < 27; ++j) {
                float p = patch[lane + 32u*j];
                s0 = fma(p, wreg0[j], s0);
                s1 = fma(p, wreg1[j], s1);
            }
            s0 = simd_sum(s0);
            s1 = simd_sum(s1);
            float per_sg_max = max(s0, s1);

            if (lane == 0) sgmax[sg] = per_sg_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Cross-sg max: have sg==0 do simd_max over the 32 values.
            if (sg == 0) {
                float v = sgmax[lane];
                v = simd_max(v);
                if (lane == 0) bcast = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            running_sum += bcast;
        }}}
    }

    if (tid == 0) {
        y[0] = running_sum * final_scale + float(Nc) * bias_val;
    }
}
