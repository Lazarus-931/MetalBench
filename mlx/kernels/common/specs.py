"""Kernel specifications.

Each entry is a plain dict consumed by mlx_helpers.load_baseline().
Keys:
    metal_function, threadgroup, input_bindings, outputs_fn,
    rtol, atol, grid_fn, scalars_fn, flops_fn, bytes_fn, BEST_FOR
Functions receive (module, inputs) and return the appropriate value.
"""

SPECS = {}


# --- Matrix ops ---

SPECS["sqr_mm"] = dict(
    metal_function="sqr_matmul_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1),
    outputs_fn=lambda mod: [dict(binding=2, dtype="f32", shape=(mod.N, mod.N))],
    rtol=1e-3, atol=1e-3,
    grid_fn=lambda mod, inputs: ((mod.N // 64) * 256, mod.N // 64, 1),
    scalars_fn=lambda mod, inputs: [dict(binding=3, dtype="u32", value=mod.N)],
    flops_fn=lambda mod, inputs: 2 * mod.N * mod.N * mod.N,
    bytes_fn=lambda mod, inputs: 3 * mod.N * mod.N * 4,
)

SPECS["rect_mm"] = dict(
    metal_function="rect_matmul_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1),
    outputs_fn=lambda mod: [dict(binding=2, dtype="f32", shape=(mod.M, mod.N))],
    rtol=1e-3, atol=1e-3,
    grid_fn=lambda mod, inputs: ((mod.N // 64) * 256, mod.M // 64, 1),
    scalars_fn=lambda mod, inputs: [
        dict(binding=3, dtype="u32", value=mod.M),
        dict(binding=4, dtype="u32", value=mod.N),
        dict(binding=5, dtype="u32", value=mod.K),
    ],
    flops_fn=lambda mod, inputs: 2 * mod.M * mod.N * mod.K,
    bytes_fn=lambda mod, inputs: 4 * (mod.M * mod.K + mod.K * mod.N + mod.M * mod.N),
)

SPECS["batch_mm"] = dict(
    metal_function="batch_matmul_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1),
    outputs_fn=lambda mod: [dict(binding=2, dtype="f32", shape=(mod.batch_size, mod.M, mod.N))],
    rtol=1e-3, atol=1e-3,
    grid_fn=lambda mod, inputs: ((mod.N // 64) * 256, (mod.M // 64) * mod.batch_size, 1),
    scalars_fn=lambda mod, inputs: [
        dict(binding=3, dtype="u32", value=mod.M),
        dict(binding=4, dtype="u32", value=mod.N),
        dict(binding=5, dtype="u32", value=mod.K),
    ],
    flops_fn=lambda mod, inputs: mod.batch_size * 2 * mod.M * mod.N * mod.K,
    bytes_fn=lambda mod, inputs: mod.batch_size * 4 * (mod.M * mod.K + mod.K * mod.N + mod.M * mod.N),
)
