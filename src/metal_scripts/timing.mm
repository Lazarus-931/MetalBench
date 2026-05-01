#import "timing.h"
#include <algorithm>

namespace metalbench {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static id<MTLCommandBuffer> dispatchOnce(
    id<MTLCommandQueue>         queue,
    id<MTLComputePipelineState> pso,
    const std::vector<Binding>& bindings,
    MTLSize                     grid,
    MTLSize                     threadgroup,
    id<MTLCounterSampleBuffer>  counterBuf = nil)
{
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    if (@available(macOS 10.15, *)) {
        if (counterBuf != nil)
            [enc sampleCountersInBuffer:counterBuf atSampleIndex:0 withBarrier:YES];
    }

    [enc setComputePipelineState:pso];
    for (const auto& b : bindings) [enc setBuffer:b.buf offset:0 atIndex:b.index];
    [enc dispatchThreads:grid threadsPerThreadgroup:threadgroup];

    if (@available(macOS 10.15, *)) {
        if (counterBuf != nil)
            [enc sampleCountersInBuffer:counterBuf atSampleIndex:1 withBarrier:YES];
    }

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return cb;
}

// ---------------------------------------------------------------------------
// Counter set enumeration
// ---------------------------------------------------------------------------

std::vector<CounterSetInfo> enumerateCounterSets(id<MTLDevice> device) {
    std::vector<CounterSetInfo> result;
    if (@available(macOS 10.15, *)) {
        NSArray<id<MTLCounterSet>>* sets = device.counterSets;
        if (!sets) return result;
        for (id<MTLCounterSet> cs in sets) {
            CounterSetInfo info;
            info.name = cs.name.UTF8String;
            for (id<MTLCounter> c in cs.counters)
                info.counters.push_back(c.name.UTF8String);
            result.push_back(std::move(info));
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Counter sampling helpers
// ---------------------------------------------------------------------------

// Find a counter set by common name substring (e.g. "Statistic").
// Returns nil if not found or API unavailable.
static id<MTLCounterSet> findCounterSet(id<MTLDevice> device, NSString* nameSubstr) {
    if (@available(macOS 10.15, *)) {
        for (id<MTLCounterSet> cs in device.counterSets) {
            if ([cs.name containsString:nameSubstr])
                return cs;
        }
    }
    return nil;
}

// Resolve a 2-sample counter buffer and populate counter_samples.
// For MTLCommonCounterSetStatistic, parses MTLCounterResultStatistic delta.
// For MTLCommonCounterSetTimestamp, parses MTLCounterResultTimestamp delta.
// Falls back to raw uint64 pairs for unknown sets.
static void resolveCounters(id<MTLCounterSampleBuffer> counterBuf,
                             const std::string& setName,
                             std::vector<std::pair<std::string, uint64_t>>& out) {
    if (@available(macOS 10.15, *)) {
        NSData* data = [counterBuf resolveCounterRange:NSMakeRange(0, 2)];
        if (!data || data.length == 0) return;

        const uint8_t* bytes = (const uint8_t*)data.bytes;
        const NSUInteger len  = data.length;

        if (setName.find("Statistic") != std::string::npos &&
            len >= 2 * sizeof(MTLCounterResultStatistic)) {
            auto* s = (const MTLCounterResultStatistic*)bytes;
            out.push_back({"vertexInvocations",
                            s[1].vertexInvocations - s[0].vertexInvocations});
            out.push_back({"fragmentInvocations",
                            s[1].fragmentInvocations - s[0].fragmentInvocations});
            out.push_back({"computeKernelInvocations",
                            s[1].computeKernelInvocations - s[0].computeKernelInvocations});
            out.push_back({"clipperInvocations",
                            s[1].clipperInvocations - s[0].clipperInvocations});
        } else if (setName.find("Timestamp") != std::string::npos &&
                   len >= 2 * sizeof(MTLCounterResultTimestamp)) {
            auto* ts = (const MTLCounterResultTimestamp*)bytes;
            // Delta in GPU ticks (device-specific frequency).
            out.push_back({"gpu_ticks_delta", ts[1].timestamp - ts[0].timestamp});
        } else {
            // Generic: emit raw uint64 pairs for each 8-byte word.
            NSUInteger nwords = len / 8;
            NSUInteger half   = nwords / 2;
            auto* u = (const uint64_t*)bytes;
            for (NSUInteger i = 0; i < half && i < 16; i++) {
                char name[32];
                snprintf(name, sizeof(name), "counter_%zu_delta", (size_t)i);
                out.push_back({name, u[half + i] - u[i]});
            }
        }
    }
}

// ---------------------------------------------------------------------------
// runTimedDispatches
// ---------------------------------------------------------------------------

TimingResult runTimedDispatches(
    id<MTLCommandQueue>         queue,
    id<MTLComputePipelineState> pso,
    const std::vector<Binding>& bindings,
    MTLSize                     grid,
    MTLSize                     threadgroup,
    int                         warmup,
    int                         iters,
    const ProfilingConfig&      prof,
    id<MTLDevice>               device)
{
    if (iters <= 0) iters = 1;

    for (int i = 0; i < warmup; i++)
        (void)dispatchOnce(queue, pso, bindings, grid, threadgroup);

    std::vector<double> times_ms;
    times_ms.reserve(iters);

    // Run all timed iterations without counter overhead.
    for (int i = 0; i < iters; i++) {
        id<MTLCommandBuffer> cb = dispatchOnce(queue, pso, bindings, grid, threadgroup);
        times_ms.push_back((cb.GPUEndTime - cb.GPUStartTime) * 1000.0);
    }

    std::sort(times_ms.begin(), times_ms.end());
    double sum = 0;
    for (double t : times_ms) sum += t;

    TimingResult result{
        .min_ms    = times_ms.front(),
        .median_ms = times_ms[times_ms.size() / 2],
        .mean_ms   = sum / static_cast<double>(times_ms.size()),
        .iters     = iters,
    };

    // Optional: one additional pass with counter sampling (does not affect
    // the timing numbers above).
    if (prof.enabled && device != nil) {
        if (@available(macOS 10.15, *)) {
            // Prefer Statistic set; fall back to Timestamp.
            id<MTLCounterSet> cs = findCounterSet(device, @"Statistic");
            if (!cs) cs = findCounterSet(device, @"Timestamp");

            if (cs) {
                MTLCounterSampleBufferDescriptor* desc =
                    [[MTLCounterSampleBufferDescriptor alloc] init];
                desc.counterSet  = cs;
                desc.sampleCount = 2;
                desc.storageMode = MTLStorageModeShared;

                NSError* err = nil;
                id<MTLCounterSampleBuffer> counterBuf =
                    [device newCounterSampleBufferWithDescriptor:desc error:&err];

                if (counterBuf && !err) {
                    (void)dispatchOnce(queue, pso, bindings, grid, threadgroup, counterBuf);
                    resolveCounters(counterBuf,
                                    std::string(cs.name.UTF8String),
                                    result.counter_samples);
                    result.counters_available  = true;
                    result.sampled_counter_set = cs.name.UTF8String;
                }
            }
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// captureDispatch
// ---------------------------------------------------------------------------

bool captureDispatch(
    id<MTLDevice>               device,
    id<MTLCommandQueue>         queue,
    id<MTLComputePipelineState> pso,
    const std::vector<Binding>& bindings,
    MTLSize                     grid,
    MTLSize                     threadgroup,
    const std::string&          tracePath)
{
    if (@available(macOS 10.15, *)) {
        MTLCaptureManager* mgr = [MTLCaptureManager sharedCaptureManager];
        MTLCaptureDescriptor* desc = [[MTLCaptureDescriptor alloc] init];
        desc.captureObject = queue;
        desc.destination   = MTLCaptureDestinationGPUTraceDocument;
        desc.outputURL     = [NSURL fileURLWithPath:@(tracePath.c_str())];

        NSError* err = nil;
        if (![mgr startCaptureWithDescriptor:desc error:&err]) {
            fprintf(stderr, "[host] capture unavailable: %s\n",
                    err ? err.localizedDescription.UTF8String : "unknown error");
            return false;
        }

        (void)dispatchOnce(queue, pso, bindings, grid, threadgroup);

        [mgr stopCapture];
        fprintf(stderr, "[host] trace written: %s\n", tracePath.c_str());
        return true;
    }
    fprintf(stderr, "[host] capture requires macOS 10.15+\n");
    return false;
}

// ---------------------------------------------------------------------------
// purgeMetalCaches
// ---------------------------------------------------------------------------

int purgeMetalCaches(std::vector<Binding>& bindings) {
    int n = 0;
    for (auto& b : bindings) {
        if (!b.buf) continue;
        [b.buf setPurgeableState:MTLPurgeableStateEmpty];
        b.buf = nil;
        n++;
    }
    bindings.clear();
    return n;
}

} // namespace metalbench
