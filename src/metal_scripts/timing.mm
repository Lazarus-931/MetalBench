#import "timing.h"
#include <algorithm>

namespace metalbench {

static id<MTLCommandBuffer> dispatchOnce(
    id<MTLCommandQueue>         queue,
    id<MTLComputePipelineState> pso,
    const std::vector<Binding>& bindings,
    MTLSize                     grid,
    MTLSize                     threadgroup)
{
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    for (const auto& b : bindings) [enc setBuffer:b.buf offset:0 atIndex:b.index];
    [enc dispatchThreads:grid threadsPerThreadgroup:threadgroup];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return cb;
}

TimingResult runTimedDispatches(
    id<MTLCommandQueue>         queue,
    id<MTLComputePipelineState> pso,
    const std::vector<Binding>& bindings,
    MTLSize                     grid,
    MTLSize                     threadgroup,
    int                         warmup,
    int                         iters)
{
    if (iters <= 0) iters = 1;

    for (int i = 0; i < warmup; i++) {
        (void)dispatchOnce(queue, pso, bindings, grid, threadgroup);
    }

    std::vector<double> times_ms;
    times_ms.reserve(iters);
    for (int i = 0; i < iters; i++) {
        id<MTLCommandBuffer> cb = dispatchOnce(queue, pso, bindings, grid, threadgroup);
        times_ms.push_back((cb.GPUEndTime - cb.GPUStartTime) * 1000.0);
    }

    std::sort(times_ms.begin(), times_ms.end());
    double sum = 0;
    for (double t : times_ms) sum += t;

    return TimingResult{
        .min_ms    = times_ms.front(),
        .median_ms = times_ms[times_ms.size() / 2],
        .mean_ms   = sum / static_cast<double>(times_ms.size()),
        .iters     = iters,
    };
}

int purgeMetalCaches(std::vector<Binding>& bindings) {
    int n = 0;
    for (auto& b : bindings) {
        if (!b.buf) continue;
        // Asks the OS to discard the backing storage. The MTLBuffer object
        // remains valid but its contents are undefined after this.
        [b.buf setPurgeableState:MTLPurgeableStateEmpty];
        b.buf = nil;
        n++;
    }
    bindings.clear();
    return n;
}

} // namespace metalbench
