// cosine_similarity: x·y / (|x| * |y|) per row. Single simd_sum reduction with threadgroup memory.
#include <metal_stdlib>
using namespace metal;

kernel void cosine_similarity_f32(
    device const float*  x   [[buffer(0)]],
    device const float*  y   [[buffer(1)]],
    device       float*  out [[buffer(2)]],
    constant     uint&   D   [[buffer(3)]],
    uint3 tid               [[thread_position_in_threadgroup]],
    uint3 tgid              [[threadgroup_position_in_grid]])
{
    const uint t = tid.x;
    const uint row = tgid.y;
    const uint off = row * D;
    const uint sg = t >> 5;

    float xi = x[off + t];
    float yi = y[off + t];
    float dot_v = xi * yi;
    float sq_x  = xi * xi;
    float sq_y  = yi * yi;

    float dot_sum  = simd_sum(dot_v);
    float sq_x_sum = simd_sum(sq_x);
    float sq_y_sum = simd_sum(sq_y);

    threadgroup float tg_dot[32], tg_sqx[32], tg_sqy[32];
    if ((t & 31) == 0) {
        tg_dot[sg] = dot_sum;
        tg_sqx[sg] = sq_x_sum;
        tg_sqy[sg] = sq_y_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t < 32) {
        float d = tg_dot[t], sx = tg_sqx[t], sy = tg_sqy[t];
        for (uint s = 16; s > 0; s >>= 1) {
            d  += simd_shuffle_down(d,  s);
            sx += simd_shuffle_down(sx, s);
            sy += simd_shuffle_down(sy, s);
        }
        if (t == 0) {
            float eps = 1e-8f;
            out[row] = d / (sqrt(sx) * sqrt(sy) + eps);
        }
    }
}
