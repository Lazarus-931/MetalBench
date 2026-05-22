#include <metal_stdlib>
using namespace metal;

static inline float gelu_erf_poly(float x) {
    const float k = 0.70710678f;
    float z = x * k;
    float t = 1.0f / (1.0f + 0.3275911f * fabs(z));
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t)
              + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f)
              * t * exp(-z * z);
    float erfz = copysign(y, z);
    return 0.5f * x * (1.0f + erfz);
}

kernel void conv3d_multi_act_bias_f32(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device       float* y [[buffer(3)]],
    constant     uint& N  [[buffer(4)]],
    constant     uint& C  [[buffer(5)]],
    constant     uint& D  [[buffer(6)]],
    constant     uint& H  [[buffer(7)]],
    constant     uint& W  [[buffer(8)]],
    constant     uint& K  [[buffer(9)]],
    constant     uint& R  [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    const uint D2 = D - R + 1, H2 = H - R + 1, W2 = W - R + 1;
    const uint total = N * D2 * H2 * W2 * K;
    for (uint idx = tid; idx < total; idx += 64 * 1024) {
        uint q = idx;
        uint k = q % K; q /= K;
        uint w2 = q % W2; q /= W2;
        uint h2 = q % H2; q /= H2;
        uint d2 = q % D2; uint n = q / D2;
        float sum = 0.0f;
        for (uint rd = 0; rd < R; ++rd)
            for (uint rh = 0; rh < R; ++rh)
                for (uint rw = 0; rw < R; ++rw)
                    for (uint c = 0; c < C; ++c)
                        sum += x[(((n * D + d2 + rd) * H + h2 + rh) * W + w2 + rw) * C + c]
                             * w[(((k * R + rd) * R + rh) * R + rw) * C + c];
        float v = fmax(sum, 0.0f);
        v = v > 0.0f ? v : 0.01f * v;
        v = gelu_erf_poly(v);
        v = 1.0f / (1.0f + exp(-v));
        y[idx] = v + b[k];
    }
}
