// Shared Metal scaffolding for MetalBench kernels.
#pragma once

#include <metal_stdlib>
using namespace metal;

inline uint idx2d(uint r, uint c, uint stride) { return r * stride + c; }

template <typename T, uint TG>
inline T tg_sum(threadgroup T* shared, uint tid, T value) {
    shared[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TG / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return shared[0];
}
