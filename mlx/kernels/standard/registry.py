"""Standard Set kernel registry — fused kernels (2+ ops in one dispatch)."""

REGISTRY = {}

# rms_norm_linear: rms_norm(x) @ W (LLaMA/Mistral core block)
REGISTRY["rms_norm_linear"] = dict(
    metal_function="rms_norm_linear_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(1024, 1024), (1024, 1024)],
    output_shape=(1024, 1024),
    rtol=1e-3, atol=1e-3,
    grid=((1024 // 64) * 256, 1024 // 64, 1),
    scalars=[
        dict(binding=3, dtype="u32", value=1024),
        dict(binding=4, dtype="u32", value=1024),
        dict(binding=5, dtype="u32", value=1024),
        dict(binding=6, dtype="f32", value=1e-5),
    ],
    flops=1024 * 1024 * 1024 * 2 + 1024 * 1024 * 5,
    bytes=1024 * 1024 * 4 * 3,
)

# silu_linear: silu(x @ W) (LLaMA FFN gate)
REGISTRY["silu_linear"] = dict(
    metal_function="silu_linear_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(1024, 1024), (1024, 1024)],
    output_shape=(1024, 1024),
    rtol=1e-3, atol=1e-3,
    grid=((1024 // 64) * 256, 1024 // 64, 1),
    scalars=[
        dict(binding=3, dtype="u32", value=1024),
        dict(binding=4, dtype="u32", value=1024),
        dict(binding=5, dtype="u32", value=1024),
    ],
    flops=1024 * 1024 * 1024 * 2 + 1024 * 1024 * 3,
    bytes=1024 * 1024 * 4 * 3,
)

# gelu_linear: gelu(x @ W) (BERT/GPT-2 FFN)
REGISTRY["gelu_linear"] = dict(
    metal_function="gelu_linear_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(1024, 1024), (1024, 1024)],
    output_shape=(1024, 1024),
    rtol=1e-3, atol=1e-3,
    grid=((1024 // 64) * 256, 1024 // 64, 1),
    scalars=[
        dict(binding=3, dtype="u32", value=1024),
        dict(binding=4, dtype="u32", value=1024),
        dict(binding=5, dtype="u32", value=1024),
    ],
    flops=1024 * 1024 * 1024 * 2 + 1024 * 1024 * 5,
    bytes=1024 * 1024 * 4 * 3,
)

# add_norm: layer_norm(x + residual) (transformer residual block)
REGISTRY["add_norm"] = dict(
    metal_function="add_norm_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(1024, 1024), (1024, 1024)],
    output_shape=(1024, 1024),
    rtol=1e-3, atol=1e-3,
    grid=(1024, 1024, 1),
    scalars=[
        dict(binding=3, dtype="u32", value=1024),
        dict(binding=4, dtype="f32", value=1e-5),
    ],
    flops=1024 * 1024 * 7,
    bytes=1024 * 1024 * 4 * 3,
)

# scaled_dot_product: softmax(Q@K^T / sqrt(d)) @ V
REGISTRY["scaled_dot_product"] = dict(
    metal_function="scaled_dot_product_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1, 2),
    input_shapes=[(128, 128), (128, 128), (128, 128)],
    output_shape=(128, 128),
    rtol=1e-3, atol=1e-3,
    grid=((128 // 64) * 256, 128 // 64, 1),
    scalars=[
        dict(binding=3, dtype="u32", value=128),
        dict(binding=4, dtype="u32", value=128),
        dict(binding=5, dtype="u32", value=128),
    ],
    flops=128 * 128 * 128 * 4 + 128 * 128 * 5,
    bytes=128 * 128 * 4 * 4,
)
