// Generic dispatcher: read JSON manifest, bind buffers, run warmup + timed
// iterations, write outputs back, print timing JSON to stdout.
#import  <Metal/Metal.h>
#import  <Foundation/Foundation.h>
#include <cstdio>
#include <cstring>
#include <vector>

#import "timing.h"
#include "setup.h"

using metalbench::Binding;
using metalbench::TimingResult;
using metalbench::ProfilingConfig;
using metalbench::CounterSetInfo;
using metalbench::runTimedDispatches;
using metalbench::enumerateCounterSets;
using metalbench::captureDispatch;
using metalbench::MChip;
using metalbench::detect_chip;

static NSString* arg(int argc, const char** argv, const char* key) {
    for (int i = 1; i + 1 < argc; i++)
        if (strcmp(argv[i], key) == 0) return @(argv[i + 1]);
    return nil;
}

static int die(const char* msg, NSError* err = nil) {
    if (err) fprintf(stderr, "metalbench_host: %s: %s\n", msg, err.localizedDescription.UTF8String);
    else     fprintf(stderr, "metalbench_host: %s\n", msg);
    return 1;
}

static MTLSize sz3(NSArray* a) {
    return MTLSizeMake([a[0] unsignedLongValue],
                       [a[1] unsignedLongValue],
                       [a[2] unsignedLongValue]);
}

// Emit a JSON array of counter set names: ["MTLCommonCounterSetStatistic",...]
static std::string counterSetsJson(const std::vector<CounterSetInfo>& sets) {
    std::string s = "[";
    for (size_t i = 0; i < sets.size(); i++) {
        if (i) s += ",";
        s += "\"";
        s += sets[i].name;
        s += "\"";
    }
    s += "]";
    return s;
}

// Emit profiling block JSON, or empty string if no counters were sampled.
static std::string profilingJson(const TimingResult& t) {
    if (!t.counters_available) {
        return "{\"counters_available\":false}";
    }
    std::string s = "{\"counters_available\":true,\"counter_set\":\"";
    s += t.sampled_counter_set;
    s += "\",\"samples\":[";
    for (size_t i = 0; i < t.counter_samples.size(); i++) {
        if (i) s += ",";
        char buf[128];
        snprintf(buf, sizeof(buf), "{\"name\":\"%s\",\"value\":%llu}",
                 t.counter_samples[i].first.c_str(),
                 (unsigned long long)t.counter_samples[i].second);
        s += buf;
    }
    s += "]}";
    return s;
}

int main(int argc, const char** argv) { @autoreleasepool {
    if (!metalbench::is_mac()) return die("not running on macOS (Darwin)");

    MChip chip = detect_chip();
    fprintf(stderr, "[host] %s (%s) | %d CPU | %d GPU cores | %.0f GB\n",
            chip.name.c_str(), metalbench::type_name(chip.type),
            chip.cpu_cores, chip.gpu_cores, chip.ram_bytes / 1e9);

    NSString* manifestPath = arg(argc, argv, "--manifest");
    if (!manifestPath) return die("missing --manifest <path>");

    NSData* mdata = [NSData dataWithContentsOfFile:manifestPath];
    if (!mdata) return die("cannot read manifest");

    NSError* err = nil;
    NSDictionary* manifest = [NSJSONSerialization JSONObjectWithData:mdata options:0 error:&err];
    if (err) return die("manifest json parse", err);

    NSString* function = manifest[@"function"];
    NSString* libPath  = manifest[@"metallib"];
    NSArray*  bspecs   = manifest[@"buffers"];
    int warmup = [manifest[@"warmup"] intValue];
    int iters  = [manifest[@"iters"]  intValue];

    ProfilingConfig prof;
    prof.enabled = [manifest[@"profile"] boolValue];

    NSString* capturePath = manifest[@"capture_path"];

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return die("no Metal device");

    // Enumerate available counter sets (zero dispatch overhead).
    std::vector<CounterSetInfo> counterSets = enumerateCounterSets(device);

    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&err];
    if (err) return die("load metallib", err);

    id<MTLFunction> func = [library newFunctionWithName:function];
    if (!func) return die([NSString stringWithFormat:@"function %@ not found", function].UTF8String);

    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func error:&err];
    if (err) return die("pipeline state", err);

    id<MTLCommandQueue> queue = [device newCommandQueue];

    fprintf(stderr,
            "[host] kernel `%s` ready | tg_mem=%lu B | max_thr/tg=%lu | setup complete\n",
            function.UTF8String,
            (unsigned long)pso.staticThreadgroupMemoryLength,
            (unsigned long)pso.maxTotalThreadsPerThreadgroup);

    std::vector<Binding> bindings;
    struct Output { id<MTLBuffer> buf; NSString* path; NSUInteger size; };
    std::vector<Output> outputs;

    for (NSDictionary* b in bspecs) {
        NSString* role = b[@"role"];
        NSUInteger idx = [b[@"binding"] unsignedIntegerValue];

        if ([role isEqualToString:@"input"]) {
            NSData* d = [NSData dataWithContentsOfFile:b[@"path"]];
            if (!d) return die([NSString stringWithFormat:@"cannot read input %@", b[@"path"]].UTF8String);
            id<MTLBuffer> buf = [device newBufferWithBytes:d.bytes length:d.length
                                                   options:MTLResourceStorageModeShared];
            bindings.push_back({buf, idx});

        } else if ([role isEqualToString:@"output"]) {
            NSUInteger sz = [b[@"size"] unsignedIntegerValue];
            id<MTLBuffer> buf = [device newBufferWithLength:sz options:MTLResourceStorageModeShared];
            bindings.push_back({buf, idx});
            outputs.push_back({buf, b[@"path"], sz});

        } else if ([role isEqualToString:@"scalar"]) {
            NSString* dt = b[@"dtype"];
            uint8_t bytes[8] = {0}; NSUInteger len = 4;
            if      ([dt isEqualToString:@"u32"]) { uint32_t v = [b[@"value"] unsignedIntValue]; memcpy(bytes, &v, 4); }
            else if ([dt isEqualToString:@"i32"]) { int32_t  v = [b[@"value"] intValue];         memcpy(bytes, &v, 4); }
            else if ([dt isEqualToString:@"f32"]) { float    v = [b[@"value"] floatValue];       memcpy(bytes, &v, 4); }
            else return die([NSString stringWithFormat:@"unknown scalar dtype %@", dt].UTF8String);
            id<MTLBuffer> buf = [device newBufferWithBytes:bytes length:len options:MTLResourceStorageModeShared];
            bindings.push_back({buf, idx});

        } else {
            return die([NSString stringWithFormat:@"unknown buffer role %@", role].UTF8String);
        }
    }

    MTLSize grid        = sz3(manifest[@"grid"]);
    MTLSize threadgroup = sz3(manifest[@"threadgroup"]);

    TimingResult t = runTimedDispatches(queue, pso, bindings, grid, threadgroup,
                                        warmup, iters, prof, device);

    for (auto& o : outputs) {
        NSData* d = [NSData dataWithBytesNoCopy:o.buf.contents length:o.size freeWhenDone:NO];
        if (![d writeToFile:o.path atomically:YES])
            return die([NSString stringWithFormat:@"cannot write output %@", o.path].UTF8String);
    }

    // Optional GPU trace capture (separate from timing; does not affect results).
    if (capturePath && capturePath.length > 0) {
        captureDispatch(device, queue, pso, bindings, grid, threadgroup,
                        std::string(capturePath.UTF8String));
    }

    if (device && chip.name == "unknown") chip.name = std::string(device.name.UTF8String);

    // JSON to stdout — internal protocol consumed by harness.py, not the user.
    // available_counter_sets is always emitted (zero overhead).
    // profiling block is always emitted so harness can check counters_available.
    printf(
      "{\"chip\":%s,\"metal_device\":\"%s\","
      "\"tg_static_mem_bytes\":%lu,\"pso_max_threads_per_tg\":%lu,"
      "\"min_ms\":%.6f,\"median_ms\":%.6f,\"mean_ms\":%.6f,\"iters\":%d,"
      "\"available_counter_sets\":%s,\"profiling\":%s}\n",
      metalbench::to_json(chip).c_str(),
      device.name.UTF8String ?: "unknown",
      (unsigned long)pso.staticThreadgroupMemoryLength,
      (unsigned long)pso.maxTotalThreadsPerThreadgroup,
      t.min_ms, t.median_ms, t.mean_ms, t.iters,
      counterSetsJson(counterSets).c_str(),
      profilingJson(t).c_str());
    return 0;
}}
