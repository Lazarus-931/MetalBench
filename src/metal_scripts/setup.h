// Apple-Silicon chip detection for MetalBench.
//
// Identifies the host (chip type, cores, RAM) so timing results can be
// partitioned per-chip. Pure C++; no Metal types — chip name comes from
// sysctl(machdep.cpu.brand_string), which already returns e.g. "Apple M2 Max"
// without needing the GPU.
//
// `bucket` is the canonical key matching `mlx_helpers.bucket_key()` on the
// Python side, so results from the C++ host and the Python harness land in
// the same `results/<bucket>/` directory.
#pragma once

#include <string>

namespace metalbench {

enum class MChipType {
    Unknown,
    M1, M1_PRO, M1_MAX, M1_ULTRA,
    M2, M2_PRO, M2_MAX, M2_ULTRA,
    M3, M3_PRO, M3_MAX, M3_ULTRA,
    M4, M4_PRO, M4_MAX,
};

struct MChip {
    MChipType   type      = MChipType::Unknown;
    std::string name      = "unknown";   // raw "Apple M2 Max"
    std::string bucket    = "unknown";   // sanitized "apple-m2-max"
    int         cpu_cores = 0;           // hw.physicalcpu
    int         gpu_cores = 0;           // best-effort; 0 if unknown
    long long   ram_bytes = 0;           // hw.memsize
};

// Inspect the host. Cheap (one sysctl per field).
MChip detect_chip();

// JSON object literal: {"type":"m2_max","name":"...","bucket":"...","cpu_cores":N,...}
std::string to_json(const MChip& chip);

const char* type_name(MChipType t);

} // namespace metalbench
