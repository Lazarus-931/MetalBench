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

const char* type_name(MChipType t) {
    switch (t) {
        case MChipType::M1:        return "m1";
        case MChipType::M1_PRO:    return "m1_pro";
        case MChipType::M1_MAX:    return "m1_max";
        case MChipType::M1_ULTRA:  return "m1_ultra";
        case MChipType::M2:        return "m2";
        case MChipType::M2_PRO:    return "m2_pro";
        case MChipType::M2_MAX:    return "m2_max";
        case MChipType::M2_ULTRA:  return "m2_ultra";
        case MChipType::M3:        return "m3";
        case MChipType::M3_PRO:    return "m3_pro";
        case MChipType::M3_MAX:    return "m3_max";
        case MChipType::M3_ULTRA:  return "m3_ultra";
        case MChipType::M4:        return "m4";
        case MChipType::M4_PRO:    return "m4_pro";
        case MChipType::M4_MAX:    return "m4_max";
        default:                   return "unknown";
    }
}

// Order matters: more-specific suffixes ("Max", "Pro", "Ultra") before bare model.
static MChipType parse_type(const std::string& name) {
    auto has = [&](const char* sub) { return name.find(sub) != std::string::npos; };
    if (has("M4 Max"))   return MChipType::M4_MAX;
    if (has("M4 Pro"))   return MChipType::M4_PRO;
    if (has("M4"))       return MChipType::M4;
    if (has("M3 Ultra")) return MChipType::M3_ULTRA;
    if (has("M3 Max"))   return MChipType::M3_MAX;
    if (has("M3 Pro"))   return MChipType::M3_PRO;
    if (has("M3"))       return MChipType::M3;
    if (has("M2 Ultra")) return MChipType::M2_ULTRA;
    if (has("M2 Max"))   return MChipType::M2_MAX;
    if (has("M2 Pro"))   return MChipType::M2_PRO;
    if (has("M2"))       return MChipType::M2;
    if (has("M1 Ultra")) return MChipType::M1_ULTRA;
    if (has("M1 Max"))   return MChipType::M1_MAX;
    if (has("M1 Pro"))   return MChipType::M1_PRO;
    if (has("M1"))       return MChipType::M1;
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
