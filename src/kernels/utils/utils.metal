// Shared Metal scaffolding for MetalBench kernels.
//
// Designed to be #include'd from any kernel in src/kernels/{common,standard,full}/.
// The Makefile passes -I src/kernels/utils so this resolves as `#include "utils.metal"`.
//
// Bar for inclusion: must be (a) reused by 2+ kernels and (b) plumbing/scaffolding,
// not compute. Activations (gelu, silu, etc.) are themselves kernel benchmarks and
// belong in their own slots, NOT here.
#pragma once

#include <metal_stdlib>
using namespace metal;

// Row-major 2D index: A[r, c] where stride is the row length.
inline uint idx2d(uint r, uint c, uint stride) { return r * stride + c; }

// Sum reduction across a 1-D threadgroup.
//   shared : threadgroup T[TG]   — caller-provided scratch
//   tid    : thread_index_in_threadgroup.x
//   TG     : threads per threadgroup (power of two)
// Returns the sum on every thread; result is broadcast via shared[0].
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
