// scaled_dot_product: softmax(Q @ K^T / sqrt(D)) @ V. One TG per query row.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint TG = 128;
constant constexpr uint MAX_M = 128;

kernel void scaled_dot_product_f32(
    device const float* Q   [[buffer(0)]],
    device const float* K   [[buffer(1)]],
    device const float* V   [[buffer(2)]],
    device       float* O   [[buffer(3)]],
    constant     uint&  M   [[buffer(4)]],
    constant     uint&  D   [[buffer(5)]],
    uint  tgid              [[threadgroup_position_in_grid]],
    uint  lid               [[thread_index_in_threadgroup]])
{
    threadgroup float scores[MAX_M];
    threadgroup float reduce_buf[TG];

    const uint row = tgid;
    if (row >= M) return;
    const float scale = rsqrt((float)D);

    // 1) scores[j] = (Q[row] . K[j]) * scale, for j in [0, M)
    for (uint j = lid; j < M; j += TG) {
        float acc = 0.0f;
        for (uint k = 0; k < D; ++k) {
            acc += Q[row * D + k] * K[j * D + k];
        }
        scores[j] = acc * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) max reduction
    float local_max = -INFINITY;
    for (uint j = lid; j < M; j += TG) local_max = max(local_max, scores[j]);
    reduce_buf[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = TG / 2; stride > 0; stride >>= 1) {
        if (lid < stride) reduce_buf[lid] = max(reduce_buf[lid], reduce_buf[lid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = reduce_buf[0];

    // 3) exponentiate and sum
    for (uint j = lid; j < M; j += TG) {
        scores[j] = exp(scores[j] - row_max);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint j = lid; j < M; j += TG) local_sum += scores[j];
    reduce_buf[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = TG / 2; stride > 0; stride >>= 1) {
        if (lid < stride) reduce_buf[lid] = reduce_buf[lid] + reduce_buf[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = reduce_buf[0];
    float inv_sum = 1.0f / row_sum;

    // 4) normalize and multiply by V: O[row, d] = sum_j scores[j] * V[j, d]
    for (uint d = lid; d < D; d += TG) {
        float acc = 0.0f;
        for (uint j = 0; j < M; ++j) {
            acc += scores[j] * V[j * D + d];
        }
        O[row * D + d] = acc * inv_sum;
    }
}
