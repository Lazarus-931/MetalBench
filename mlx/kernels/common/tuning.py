# Per-chip dispatch-time overrides for MetalBench kernels.
#
# Key: ("kernel_name", "chip_type") -> {"threadgroup": (x,y,z), "grid": (x,y,z)}
# Only keys that differ from the registry default need to be listed.
# Fallback chain: exact chip type ("m4_max") -> generation ("m4") -> registry default.
#
# NOTE: tile constants compiled into .metallib (BM, BN, BK) cannot be changed
# here — those require MTLFunctionConstantValues specialisation (future work).
# This table controls dispatch-time params only: threadgroup size and grid dims.
#
# Pro/Max/Ultra variants within a generation differ in GPU core count but not
# in threadgroup memory limits or register pressure, so threadgroup size rarely
# needs to change between variants. Grid size (number of threadgroups) is the
# primary lever for feeding more GPU cores on Max/Ultra chips.
#
# Example entry (uncomment and tune on actual hardware):
#   ("relu", "m4_max"): {"grid": (256 * 1024, 1, 1)},  # 4x default for 40-core GPU

CHIP_TUNING: dict[tuple[str, str], dict] = {}
