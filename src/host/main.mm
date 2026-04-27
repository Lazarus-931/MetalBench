// MetalBench host runner.
//
// Generic dispatcher: takes a JSON manifest describing one kernel launch,
// loads the metallib, binds buffers/scalars, runs warmup + timed iterations,
// writes outputs back to disk, and prints timing JSON to stdout.
//
// Manifest schema (see python/metalbench/host.py for the writer):
//   {
//     "function":    "kernel_name",
//     "metallib":    "/abs/path/to/x.metallib",
//     "buffers": [
//       {"binding": 0, "role": "input",  "path": "/tmp/in0.bin"},
//       {"binding": 1, "role": "output", "path": "/tmp/out0.bin", "size": 4096},
//       {"binding": 2, "role": "scalar", "dtype": "u32", "value": 1024}
//     ],
//     "grid":        [N, 1, 1],
//     "threadgroup": [64, 1, 1],
//     "warmup":      5,
//     "iters":       50
//   }
//
// Stdout (last line):
//   {"min_ms":..,"median_ms":..,"mean_ms":..,"iters":..}

#import  <Metal/Metal.h>
#import  <Foundation/Foundation.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static NSString* argValue(int argc, const char** argv, const char* key) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], key) == 0) return @(argv[i + 1]);
    }
    return nil;
}

static int die(const char* msg, NSError* err = nil) {
    if (err) fprintf(stderr, "metalbench_host: %s: %s\n", msg, err.localizedDescription.UTF8String);
    else     fprintf(stderr, "metalbench_host: %s\n", msg);
    return 1;
}

static MTLSize sizeFromArray(NSArray* a) {
    return MTLSizeMake([a[0] unsignedLongValue],
                       [a[1] unsignedLongValue],
                       [a[2] unsignedLongValue]);
}

int main(int argc, const char** argv) { @autoreleasepool {
    NSString* manifestPath = argValue(argc, argv, "--manifest");
    if (!manifestPath) return die("missing --manifest <path>");

    NSData* mdata = [NSData dataWithContentsOfFile:manifestPath];
    if (!mdata) return die("cannot read manifest");

    NSError* err = nil;
    NSDictionary* manifest = [NSJSONSerialization JSONObjectWithData:mdata options:0 error:&err];
    if (err) return die("manifest json parse", err);

    NSString* function = manifest[@"function"];
    NSString* libPath  = manifest[@"metallib"];
    NSArray*  bspecs   = manifest[@"buffers"];
    NSArray*  gridA    = manifest[@"grid"];
    NSArray*  tgA      = manifest[@"threadgroup"];
    int warmup = [manifest[@"warmup"] intValue];
    int iters  = [manifest[@"iters"]  intValue];
    if (iters <= 0) iters = 1;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return die("no Metal device");

    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&err];
    if (err) return die("load metallib", err);

    id<MTLFunction> func = [library newFunctionWithName:function];
    if (!func) return die([NSString stringWithFormat:@"function %@ not found in metallib", function].UTF8String);

    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func error:&err];
    if (err) return die("pipeline state", err);

    id<MTLCommandQueue> queue = [device newCommandQueue];

    // Bindings.
    struct Binding { id<MTLBuffer> buf; NSUInteger index; };
    std::vector<Binding> bindings;

    // Outputs to read back after the timed run.
    struct Output { id<MTLBuffer> buf; NSString* path; NSUInteger size; };
    std::vector<Output> outputs;

    for (NSDictionary* b in bspecs) {
        NSString* role = b[@"role"];
        NSUInteger idx = [b[@"binding"] unsignedIntegerValue];

        if ([role isEqualToString:@"input"]) {
            NSData* d = [NSData dataWithContentsOfFile:b[@"path"]];
            if (!d) return die([NSString stringWithFormat:@"cannot read input %@", b[@"path"]].UTF8String);
            id<MTLBuffer> buf = [device newBufferWithBytes:d.bytes
                                                    length:d.length
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

    MTLSize grid = sizeFromArray(gridA);
    MTLSize tg   = sizeFromArray(tgA);

    auto dispatchOnce = ^id<MTLCommandBuffer>() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        for (auto& bnd : bindings) [enc setBuffer:bnd.buf offset:0 atIndex:bnd.index];
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return cb;
    };

    for (int i = 0; i < warmup; i++) (void)dispatchOnce();

    std::vector<double> times_ms; times_ms.reserve(iters);
    for (int i = 0; i < iters; i++) {
        id<MTLCommandBuffer> cb = dispatchOnce();
        times_ms.push_back((cb.GPUEndTime - cb.GPUStartTime) * 1000.0);
    }

    // Read outputs back.
    for (auto& o : outputs) {
        NSData* d = [NSData dataWithBytesNoCopy:o.buf.contents length:o.size freeWhenDone:NO];
        if (![d writeToFile:o.path atomically:YES])
            return die([NSString stringWithFormat:@"cannot write output %@", o.path].UTF8String);
    }

    std::sort(times_ms.begin(), times_ms.end());
    double minv = times_ms.front();
    double medv = times_ms[times_ms.size() / 2];
    double sum  = 0; for (double t : times_ms) sum += t;
    double mean = sum / times_ms.size();

    printf("{\"min_ms\":%.6f,\"median_ms\":%.6f,\"mean_ms\":%.6f,\"iters\":%d}\n",
           minv, medv, mean, iters);
    return 0;
}}
