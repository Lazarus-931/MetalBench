"""Kernel specifications consumed by mlx_helpers.load_baseline()."""

SPECS = {}

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
    shapes=dict(N=1024),
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
    shapes=dict(M=1024, K=4096, N=2048),
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
    shapes=dict(batch_size=128, M=128, K=256, N=512),
)

SPECS["relu"] = dict(
    metal_function="relu_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0,),
    outputs_fn=lambda mod: [dict(binding=1, dtype="f32", shape=(mod.batch_size, mod.dim))],
    rtol=0, atol=0,
    grid_fn=lambda mod, inputs: (64 * 1024, 1, 1),
    scalars_fn=lambda mod, inputs: [
        dict(binding=2, dtype="u32", value=mod.batch_size * mod.dim),
        dict(binding=3, dtype="u32", value=64 * 1024),
    ],
    flops_fn=lambda mod, inputs: mod.batch_size * mod.dim,
    bytes_fn=lambda mod, inputs: 2 * mod.batch_size * mod.dim * 4,
    shapes=dict(batch_size=16, dim=16384),
)

SPECS["sigmoid"] = dict(
    metal_function="sigmoid_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0,),
    outputs_fn=lambda mod: [dict(binding=1, dtype="f32", shape=(16, 16384))],
    rtol=1e-3, atol=1e-3,
    grid_fn=lambda mod, inputs: (64 * 1024, 1, 1),
    scalars_fn=lambda mod, inputs: [
        dict(binding=2, dtype="u32", value=16 * 16384),
        dict(binding=3, dtype="u32", value=64 * 1024),
    ],
    flops_fn=lambda mod, inputs: 16 * 16384 * 4,
    bytes_fn=lambda mod, inputs: 2 * 16 * 16384 * 4,
)

SPECS["leaky_relu"] = dict(
    metal_function="leaky_relu_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0,),
    outputs_fn=lambda mod: [dict(binding=1, dtype="f32", shape=(16, 16384))],
    rtol=1e-3, atol=1e-3,
    grid_fn=lambda mod, inputs: (64 * 1024, 1, 1),
    scalars_fn=lambda mod, inputs: [
        dict(binding=2, dtype="u32", value=16 * 16384),
        dict(binding=3, dtype="u32", value=64 * 1024),
        dict(binding=4, dtype="f32", value=0.01),
    ],
    flops_fn=lambda mod, inputs: 16 * 16384 * 3,
    bytes_fn=lambda mod, inputs: 2 * 16 * 16384 * 4,
)

SPECS["swish"] = dict(
    metal_function="swish_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0,),
    outputs_fn=lambda mod: [dict(binding=1, dtype="f32", shape=(16, 16384))],
    rtol=1e-3, atol=1e-3,
    grid_fn=lambda mod, inputs: (64 * 1024, 1, 1),
    scalars_fn=lambda mod, inputs: [
        dict(binding=2, dtype="u32", value=16 * 16384),
        dict(binding=3, dtype="u32", value=64 * 1024),
    ],
    flops_fn=lambda mod, inputs: 16 * 16384 * 5,
    bytes_fn=lambda mod, inputs: 2 * 16 * 16384 * 4,
)

SPECS["layer_norm"] = dict(
    metal_function="layer_norm_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0,),
    outputs_fn=lambda mod: [dict(binding=1, dtype="f32", shape=(1024, 1024))],
    rtol=1e-3, atol=1e-3,
    grid_fn=lambda mod, inputs: (1024, 1024, 1),
    scalars_fn=lambda mod, inputs: [
        dict(binding=2, dtype="u32", value=1024),
        dict(binding=3, dtype="f32", value=1e-5),
    ],
    flops_fn=lambda mod, inputs: 1024 * 1024 * 7,
    bytes_fn=lambda mod, inputs: 2 * 1024 * 1024 * 4,
)
