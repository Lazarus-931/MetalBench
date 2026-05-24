#include "setup.h"
#include <sys/sysctl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>

namespace metalbench {

static std::string sysctl_str(const char* key) {
    size_t len = 0;
    if (sysctlbyname(key, nullptr, &len, nullptr, (size_t)0) != 0 || len == 0) return {};
    std::string buf(len, '\0');
    if (sysctlbyname(key, buf.data(), &len, nullptr, (size_t)0) != 0) return {};
    while (!buf.empty() && (buf.back() == '\0' || buf.back() == '\n')) buf.pop_back();
    return buf;
}

static long long sysctl_int(const char* key) {
    long long v = 0;
    size_t len = sizeof(v);
    if (sysctlbyname(key, &v, &len, nullptr, (size_t)0) != 0) return 0;
    return v;
}

// type_name() and parse_type() are both generated from the same X-macro
// (chip_table.h) so they can never drift out of sync. To add an M6, edit
// chips.json and rebuild — nothing in this file changes.

const char* type_name(MChipType t) {
    switch (t) {
#define METALBENCH_X_CASE(EnumTag, Name, Needle) \
        case MChipType::EnumTag: return Name;
        METALBENCH_CHIP_LIST(METALBENCH_X_CASE)
#undef METALBENCH_X_CASE
        default: return "unknown";
    }
}

// Order matters: chip_table.h emits variants most-specific-first within a
// generation and newest-generation-first across generations, so a simple
// top-down `find()` ladder gives correct matches.
static MChipType parse_type(const std::string& name) {
#define METALBENCH_X_PARSE(EnumTag, Name, Needle) \
        if (name.find(Needle) != std::string::npos) return MChipType::EnumTag;
    METALBENCH_CHIP_LIST(METALBENCH_X_PARSE)
#undef METALBENCH_X_PARSE
    return MChipType::Unknown;
}

// Must match Python `mlx_helpers.bucket_key()`: lowercase, spaces → hyphens,
// only [a-z0-9-] retained.
static std::string make_bucket(const std::string& name) {
    std::string s; s.reserve(name.size());
    for (char c : name) {
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        if (c == ' ' || c == '_') c = '-';
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-') {
            s.push_back(c);
        }
    }
    while (!s.empty() && s.back()  == '-') s.pop_back();
    while (!s.empty() && s.front() == '-') s.erase(s.begin());
    return s.empty() ? "unknown" : s;
}

bool is_mac() {
    std::string os = sysctl_str("kern.ostype");
    return os == "Darwin";
}

// Parse `ioreg -lr -k gpu-core-count` for the first integer following
// "gpu-core-count" = . Returns 0 if unavailable.
static int detect_gpu_cores() {
    FILE* p = popen("ioreg -lr -k gpu-core-count 2>/dev/null", "r");
    if (!p) return 0;
    std::string out;
    char buf[256];
    while (fgets(buf, sizeof(buf), p)) out.append(buf);
    pclose(p);
    std::smatch m;
    std::regex re("gpu-core-count\"\\s*=\\s*(\\d+)");
    if (std::regex_search(out, m, re) && m.size() >= 2) {
        return std::atoi(m[1].str().c_str());
    }
    return 0;
}

MChip detect_chip() {
    MChip chip;
    chip.name      = sysctl_str("machdep.cpu.brand_string");
    if (chip.name.empty()) chip.name = "unknown";
    chip.type      = parse_type(chip.name);
    chip.bucket    = make_bucket(chip.name);
    chip.cpu_cores = (int)sysctl_int("hw.physicalcpu");
    chip.gpu_cores = detect_gpu_cores();
    chip.ram_bytes = sysctl_int("hw.memsize");
    return chip;
}

std::string to_json(const MChip& chip) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\"type\":\"%s\",\"name\":\"%s\",\"bucket\":\"%s\","
        "\"cpu_cores\":%d,\"gpu_cores\":%d,\"ram_bytes\":%lld}",
        type_name(chip.type), chip.name.c_str(), chip.bucket.c_str(),
        chip.cpu_cores, chip.gpu_cores, chip.ram_bytes);
    return buf;
}

} // namespace metalbench
