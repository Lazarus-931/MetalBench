// Naive scaffold: conv3d → softmax along C → maxpool 2x2x2 → maxpool 2x2x2.
#include <metal_stdlib>
using namespace metal;

kernel void conv3d_softmax_pool_f32(
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
    uint tid [[thread_position_in_grid]])
{
    const uint D_conv = D - R + 1;   // 30
    const uint H_conv = H - R + 1;   // 30
    const uint W_conv = W - R + 1;
    const uint D_out = (D_conv / 2) / 2;   // 7
    const uint H_out = (H_conv / 2) / 2;
    const uint W_out = (W_conv / 2) / 2;
    const uint total = N * D_out * H_out * W_out * K;

    for (uint idx = tid; idx < total; idx += 64 * 1024) {
        uint q = idx;
        uint k = q % K; q /= K;
        uint w4 = q % W_out; q /= W_out;
        uint h4 = q % H_out; q /= H_out;
        uint d4 = q % D_out; uint n = q / D_out;

        float best = -INFINITY;
        for (uint dd = 0; dd < 4; ++dd)
            for (uint hh = 0; hh < 4; ++hh)
                for (uint ww = 0; ww < 4; ++ww) {
                    uint dc = d4 * 4 + dd;
                    uint hc = h4 * 4 + hh;
                    uint wc = w4 * 4 + ww;
                    if (dc >= D_conv || hc >= H_conv || wc >= W_conv) continue;
                    float conv_outs[64];
                    float max_v = -INFINITY;
                    for (uint kk = 0; kk < K; ++kk) {
                        float s = 0.0f;
                        for (uint rd = 0; rd < R; ++rd)
                            for (uint rh = 0; rh < R; ++rh)
                                for (uint rw = 0; rw < R; ++rw)
                                    for (uint c = 0; c < C; ++c)
                                        s += x[(((n * D + dc + rd) * H + hc + rh) * W + wc + rw) * C + c]
                                           * w[(((kk * R + rd) * R + rh) * R + rw) * C + c];
                        conv_outs[kk] = s;
                        if (s > max_v) max_v = s;
                    }
                    float denom = 0.0f;
                    for (uint kk = 0; kk < K; ++kk) {
                        float e = exp(conv_outs[kk] - max_v);
                        conv_outs[kk] = e;
                        denom += e;
                    }
                    float val = conv_outs[k] / denom;
                    if (val > best) best = val;
                }
        y[idx] = best;
    }
}
