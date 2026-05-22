// embedding: out[s, d] = table[(uint)indices[s], d]. (S,) × (V, D) → (S, D).
#include <metal_stdlib>
using namespace metal;

kernel void embedding_f32(
    device const float*  indices  [[buffer(0)]],
    device const float*  table    [[buffer(1)]],
    device       float*  y        [[buffer(2)]],
    constant     uint&   S        [[buffer(3)]],
    constant     uint&   V        [[buffer(4)]],
    constant     uint&   D        [[buffer(5)]],
    uint tid                     [[thread_position_in_grid]])
{
    const uint total = S * D;
    const uint grid_size = 64 * 1024;
    for (uint i = tid; i < total; i += grid_size) {
        uint s = i / D;
        uint d = i % D;
        uint idx = clamp(uint(indices[s]), 0u, V - 1u);
        y[i] = table[idx * D + d];
    }
}
