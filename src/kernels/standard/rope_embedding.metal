// rope_embedding: rotary position embedding. Input (S, D), D even.
#include <metal_stdlib>
using namespace metal;

kernel void rope_embedding_f32(
    device const float*  x         [[buffer(0)]],
    device       float*  y         [[buffer(1)]],
    constant     uint&   S         [[buffer(2)]],
    constant     uint&   D         [[buffer(3)]],
    constant     float&  base      [[buffer(4)]],
    uint  tid                     [[thread_position_in_grid]])
{
    const uint half_d = D / 2;
    const uint total  = S * half_d;
    if (tid >= total) return;

    const uint s = tid / half_d;
    const uint i = tid % half_d;
    float omega = 1.0f / pow(base, (2.0f * float(i)) / float(D));
    float ang = float(s) * omega;
    float c = cos(ang), sn = sin(ang);

    float x0 = x[s * D + 2 * i];
    float x1 = x[s * D + 2 * i + 1];
    y[s * D + 2 * i]     = x0 * c - x1 * sn;
    y[s * D + 2 * i + 1] = x0 * sn + x1 * c;
}
