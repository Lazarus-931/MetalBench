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

// chip_table.h is auto-generated from /chips.json by
// scripts/generate_chip_table.py (invoked by the Makefile). To add a new
// chip generation (M6, M7, ...) edit chips.json — never edit chip_table.h.
#include "chip_table.h"

namespace metalbench {

// Build the MChipType enum from the X-macro defined in chip_table.h. Adding
// a chip to chips.json adds new entries here automatically.
#define METALBENCH_X_ENUM(EnumTag, Name, Needle) EnumTag,
enum class MChipType {
    Unknown,
    METALBENCH_CHIP_LIST(METALBENCH_X_ENUM)
};
#undef METALBENCH_X_ENUM

struct MChip {
    MChipType   type      = MChipType::Unknown;
    std::string name      = "unknown";   // raw "Apple M2 Max"
    std::string bucket    = "unknown";   // sanitized "apple-m2-max"
    int         cpu_cores = 0;           // hw.physicalcpu
    int         gpu_cores = 0;           // ioreg gpu-core-count, 0 if unknown
    long long   ram_bytes = 0;           // hw.memsize
};

// Cheap macOS check (sysctl kern.ostype == "Darwin"). The host binary
// cannot link Metal.framework on non-Mac platforms anyway, but this is
// defense-in-depth + a clear fail-fast message.
bool is_mac();

// Inspect the host. Sysctl + one ioreg shell-out for GPU core count (~10ms).
MChip detect_chip();

// JSON object literal: {"type":"m2_max","name":"...","bucket":"...","cpu_cores":N,...}
std::string to_json(const MChip& chip);

const char* type_name(MChipType t);

} // namespace metalbench
