# Per-chip dispatch overrides. Key: (kernel, chip) -> {threadgroup, grid}.
# Fallback: exact chip -> generation -> registry default.

CHIP_TUNING: dict[tuple[str, str], dict] = {
    ("softmax", "m4"): {"threadgroup": (256, 1, 1), "grid": (256, 1024, 1)},
}
