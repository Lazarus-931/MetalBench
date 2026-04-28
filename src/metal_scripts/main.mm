// MetalBench host runner.
//
// Generic dispatcher: takes a JSON manifest describing one kernel launch,
// loads the metallib, binds buffers/scalars, runs warmup + timed iterations,
// writes outputs back to disk, and prints timing JSON to stdout. The actual
// dispatch loop and Metal cache helpers live in timing.{h,mm}.
//
// Manifest schema:
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
//   {"device":"...","min_ms":..,"median_ms":..,"mean_ms":..,"iters":..}

#import  <Metal/Metal.h>
#import  <Foundation/Foundation.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#import "timing.h"
#include "setup.h"

using metalbench::Binding;
using metalbench::TimingResult;
using metalbench::runTimedDispatches;
using metalbench::MChip;
using metalbench::detect_chip;

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

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return die("no Metal device");

    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&err];
    if (err) return die("load metallib", err);

    id<MTLFunction> func = [library newFunctionWithName:function];
    if (!func) return die([NSString stringWithFormat:@"function %@ not found in metallib", function].UTF8String);

    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func error:&err];
    if (err) return die("pipeline state", err);

    id<MTLCommandQueue> queue = [device newCommandQueue];

    std::vector<Binding> bindings;

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

    TimingResult t = runTimedDispatches(queue, pso, bindings, grid, tg, warmup, iters);

    for (auto& o : outputs) {
        NSData* d = [NSData dataWithBytesNoCopy:o.buf.contents length:o.size freeWhenDone:NO];
        if (![d writeToFile:o.path atomically:YES])
            return die([NSString stringWithFormat:@"cannot write output %@", o.path].UTF8String);
    }

    MChip chip = detect_chip();
    if (chip.gpu_cores == 0 && device) {
        // MTLDevice can't directly report core count, but its name is the
        // same string sysctl returned — keep them aligned.
        chip.name = std::string(device.name.UTF8String);
    }
    printf("{\"chip\":%s,\"metal_device\":\"%s\","
           "\"min_ms\":%.6f,\"median_ms\":%.6f,\"mean_ms\":%.6f,\"iters\":%d}\n",
           metalbench::to_json(chip).c_str(),
           device.name.UTF8String ?: "unknown",
           t.min_ms, t.median_ms, t.mean_ms, t.iters);
    return 0;
}}
