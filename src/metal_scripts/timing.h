// Metal-side timing + cache helpers for the MetalBench host.
//
// `runTimedDispatches` owns the warmup + timed dispatch loop and returns
// per-iteration GPU time in milliseconds (computed from
// MTLCommandBuffer.GPUEndTime - GPUStartTime, which is GPU-side, not
// wall-clock).
//
// `purgeMetalCaches` is the equivalent of `mx.metal.clear_cache()`. Metal
// itself doesn't expose a "clear all caches" call — what we *can* control
// is our own MTLBuffer/MTLComputePipelineState retention, plus marking
// resources purgeable so the OS may discard their backing storage. To get
// a clean cold-start measurement, callers should:
//   1. call purgeMetalCaches(bindings)         → drops buffer storage
//   2. discard their MTLComputePipelineState   → drops the cached PSO
//   3. recreate both                           → forces re-load + re-compile
//      from the on-disk metallib + shader cache
#pragma once

#import  <Metal/Metal.h>
#include <vector>

namespace metalbench {

struct Binding {
    id<MTLBuffer> buf;
    NSUInteger    index;
};

struct TimingResult {
    double min_ms;
    double median_ms;
    double mean_ms;
    int    iters;
};

TimingResult runTimedDispatches(
    id<MTLCommandQueue>         queue,
    id<MTLComputePipelineState> pso,
    const std::vector<Binding>& bindings,
    MTLSize                     grid,
    MTLSize                     threadgroup,
    int                         warmup,
    int                         iters);

// Mark every buffer in `bindings` as purgeable=Empty (asks the OS to drop
// the backing storage) and clears the vector. Returns count purged.
//
// IMPORTANT: buffers must not be used after this call. Re-create them from
// source data if you intend to redispatch.
int purgeMetalCaches(std::vector<Binding>& bindings);

} // namespace metalbench
