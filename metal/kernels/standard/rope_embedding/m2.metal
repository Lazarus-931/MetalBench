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
    const uint half_d = D >> 1;
    const uint total  = S * half_d;
    if (tid >= total) return;

    const uint s = tid / half_d;
    const uint i = tid - s * half_d;

    float omega = 1.0f / pow(base, (2.0f * float(i)) / float(D));
    float ang = float(s) * omega;
    float c = precise::cos(ang);
    float sn = precise::sin(ang);

    const uint off = s * D + (i << 1);
    float2 xv = *((device const float2*)(x + off));
    float2 yv;
    yv.x = fma(xv.x, c, -xv.y * sn);
    yv.y = fma(xv.x, sn,  xv.y * c);
    *((device float2*)(y + off)) = yv;
}
