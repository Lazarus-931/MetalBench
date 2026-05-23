// Metal-side timing + profiling helpers for MetalBench host binary.
#pragma once

#import  <Metal/Metal.h>
#include <string>
#include <vector>

namespace metalbench {

struct Binding {
    id<MTLBuffer> buf;
    NSUInteger    index;
};

struct CounterSetInfo {
    std::string name;
    std::vector<std::string> counters;
};

struct ProfilingConfig {
    bool enabled = false;
};

struct TimingResult {
    double min_ms;
    double median_ms;
    double mean_ms;
    int    iters;
    // Populated only when ProfilingConfig::enabled and counters are available.
    bool        counters_available = false;
    std::string sampled_counter_set;
    std::vector<std::pair<std::string, uint64_t>> counter_samples;
};

// Enumerate available MTLCounterSets on `device`. Zero dispatch overhead.
// Returns empty vector on macOS < 10.15 or if device.counterSets is nil.
std::vector<CounterSetInfo> enumerateCounterSets(id<MTLDevice> device);

TimingResult runTimedDispatches(
    id<MTLCommandQueue>         queue,
    id<MTLComputePipelineState> pso,
    const std::vector<Binding>& bindings,
    MTLSize                     grid,
    MTLSize                     threadgroup,
    int                         warmup,
    int                         iters,
    const ProfilingConfig&      prof   = {},
    id<MTLDevice>               device = nil);

// Record a single dispatch to `tracePath` (.gputrace) using MTLCaptureManager.
// Returns true on success. Silently returns false if capture is unavailable
// (e.g., on hardened-runtime builds without the gpu-capture entitlement).
// The trace is separate from and does not affect the timed measurements.
bool captureDispatch(
    id<MTLDevice>               device,
    id<MTLCommandQueue>         queue,
    id<MTLComputePipelineState> pso,
    const std::vector<Binding>& bindings,
    MTLSize                     grid,
    MTLSize                     threadgroup,
    const std::string&          tracePath);

// Mark every buffer in `bindings` as purgeable=Empty (asks the OS to drop
// the backing storage) and clears the vector. Returns count purged.
//
// IMPORTANT: buffers must not be used after this call. Re-create them from
// source data if you intend to redispatch.
int purgeMetalCaches(std::vector<Binding>& bindings);

} // namespace metalbench
