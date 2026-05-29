"""Standard Set kernel registry — fused kernels (2+ ops in one dispatch).

NOTE: This module also imports/declares Full Set kernels for now (they live in
mlx/kernels/full/ + metal/kernels/full/). The harness loader checks all three
sets (common / standard / full).
"""

REGISTRY = {}

REGISTRY["matmul_gelu_softmax"] = dict(
    metal_function="matmul_gelu_softmax_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(256, 256), (256, 256)],
    output_shape=(256, 256),
    rtol=1e-3, atol=1e-3,
    grid=(256, 256, 1),
    scalars=[dict(binding=3, dtype="u32", value=256),
             dict(binding=4, dtype="u32", value=256),
             dict(binding=5, dtype="u32", value=256)],
    flops=256 * (2*256*256 + 256*8 + 256*4),
    bytes=4 * (256*256*2 + 256*256),
)

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

# ---------------------------------------------------------------------------
# Planned set: rope, swiglu, softmax_attention, residual_add,
#                  dropout, instance_norm, group_norm, cross_entropy_loss
# ---------------------------------------------------------------------------

# rope_embedding: rotate (S, D) tensor. Grid = S * D/2 threads.
REGISTRY["rope_embedding"] = dict(
    metal_function="rope_embedding_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0,),
    input_shapes=[(128, 64)],
    output_shape=(128, 64),
    rtol=1e-4, atol=1e-5,
    grid=(((128 * 32) + 255) // 256 * 256, 1, 1),
    scalars=[dict(binding=2, dtype="u32", value=128),
             dict(binding=3, dtype="u32", value=64),
             dict(binding=4, dtype="f32", value=10000.0)],
    flops=128 * 32 * 6,
    bytes=128 * 64 * 4 * 2,
)

# swiglu: silu(x @ Wg) * (x @ Wu). Shapes (M, K) (K, N) (K, N).
REGISTRY["swiglu"] = dict(
    metal_function="swiglu_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1, 2),
    input_shapes=[(256, 256), (256, 256), (256, 256)],
    output_shape=(256, 256),
    rtol=1e-3, atol=1e-3,
    grid=(((256 * 256) + 255) // 256 * 256, 1, 1),
    scalars=[dict(binding=4, dtype="u32", value=256),
             dict(binding=5, dtype="u32", value=256),
             dict(binding=6, dtype="u32", value=256)],
    flops=256 * 256 * 256 * 4 + 256 * 256 * 5,
    bytes=4 * (256 * 256 * 3 + 256 * 256),
)

# softmax_attention: full (Q,K,V) → out. Shapes (S, D).
REGISTRY["softmax_attention"] = dict(
    metal_function="softmax_attention_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0, 1, 2),
    input_shapes=[(128, 64), (128, 64), (128, 64)],
    output_shape=(128, 64),
    rtol=1e-3, atol=1e-3,
    grid=(1024, 128, 1),
    scalars=[dict(binding=4, dtype="u32", value=128),
             dict(binding=5, dtype="u32", value=64)],
    flops=128 * (128 * 64 * 2 + 128 * 5 + 128 * 64 * 2),
    bytes=4 * (128 * 64 * 3 + 128 * 64),
)

# residual_add: x + alpha * residual. Element-wise.
REGISTRY["residual_add"] = dict(
    metal_function="residual_add_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(1024, 1024), (1024, 1024)],
    output_shape=(1024, 1024),
    rtol=1e-5, atol=1e-6,
    grid=(64 * 1024, 1, 1),
    scalars=[dict(binding=3, dtype="u32", value=1024 * 1024),
             dict(binding=4, dtype="u32", value=64 * 1024),
             dict(binding=5, dtype="f32", value=1.0)],
    flops=1024 * 1024 * 2,
    bytes=4 * 1024 * 1024 * 3,
)

# dropout: x * mask / (1-p).
REGISTRY["dropout"] = dict(
    metal_function="dropout_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(1024, 1024), (1024, 1024)],
    output_shape=(1024, 1024),
    rtol=1e-5, atol=1e-6,
    grid=(64 * 1024, 1, 1),
    scalars=[dict(binding=3, dtype="u32", value=1024 * 1024),
             dict(binding=4, dtype="u32", value=64 * 1024),
             dict(binding=5, dtype="f32", value=0.1)],
    flops=1024 * 1024 * 2,
    bytes=4 * 1024 * 1024 * 3,
)

# instance_norm: (N, C, H, W). One TG per (n, c). N*C = 8*64 = 512 TGs.
REGISTRY["instance_norm"] = dict(
    metal_function="instance_norm_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0,),
    input_shapes=[(8, 64, 32, 32)],
    output_shape=(8, 64, 32, 32),
    rtol=1e-3, atol=1e-3,
    grid=(1024, 8 * 64, 1),
    scalars=[dict(binding=2, dtype="u32", value=8),
             dict(binding=3, dtype="u32", value=64),
             dict(binding=4, dtype="u32", value=32),
             dict(binding=5, dtype="u32", value=32),
             dict(binding=6, dtype="f32", value=1e-5)],
    flops=8 * 64 * 32 * 32 * 8,
    bytes=4 * 8 * 64 * 32 * 32 * 2,
)

# group_norm: (N, C, H, W) with G groups. N*G = 8*8 = 64 TGs.
REGISTRY["group_norm"] = dict(
    metal_function="group_norm_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0,),
    input_shapes=[(8, 64, 32, 32)],
    output_shape=(8, 64, 32, 32),
    rtol=1e-3, atol=1e-3,
    grid=(1024, 8 * 8, 1),
    scalars=[dict(binding=2, dtype="u32", value=8),
             dict(binding=3, dtype="u32", value=64),
             dict(binding=4, dtype="u32", value=32),
             dict(binding=5, dtype="u32", value=32),
             dict(binding=6, dtype="u32", value=8),
             dict(binding=7, dtype="f32", value=1e-5)],
    flops=8 * 64 * 32 * 32 * 8,
    bytes=4 * 8 * 64 * 32 * 32 * 2,
)

# cross_entropy_loss: per-row (N,). (N, C) logits + (N, C) one-hot.
REGISTRY["cross_entropy_loss"] = dict(
    metal_function="cross_entropy_loss_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(1024, 1024), (1024, 1024)],
    output_shape=(1024,),
    rtol=1e-3, atol=1e-3,
    grid=(1024, 1024, 1),
    scalars=[dict(binding=3, dtype="u32", value=1024),
             dict(binding=4, dtype="u32", value=1024)],
    flops=1024 * 1024 * 6,
    bytes=4 * 1024 * 1024 * 2 + 1024 * 4,
)

# ---------------------------------------------------------------------------
# New set (11): log_softmax, masked_softmax, bias_add, fused_add_rms_norm,
#               linear_bias, bias_gelu, fused_qkv_projection, attention_scores,
#               nll_loss, log_softmax_cross_entropy
# ---------------------------------------------------------------------------

REGISTRY["log_softmax"] = dict(
    metal_function="log_softmax_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0,),
    input_shapes=[(1024, 1024)],
    output_shape=(1024, 1024),
    rtol=1e-4, atol=1e-4,
    grid=(1024, 1024, 1),
    scalars=[dict(binding=2, dtype="u32", value=1024)],
    flops=1024 * 1024 * 5,
    bytes=4 * 1024 * 1024 * 2,
)

REGISTRY["masked_softmax"] = dict(
    metal_function="masked_softmax_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(1024, 1024), (1024, 1024)],
    output_shape=(1024, 1024),
    rtol=1e-4, atol=1e-4,
    grid=(1024, 1024, 1),
    scalars=[dict(binding=3, dtype="u32", value=1024)],
    flops=1024 * 1024 * 6,
    bytes=4 * 1024 * 1024 * 3,
)

REGISTRY["bias_add"] = dict(
    metal_function="bias_add_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(1024, 1024), (1024,)],
    output_shape=(1024, 1024),
    rtol=1e-5, atol=1e-6,
    grid=(8 * 1024, 1, 1),
    scalars=[dict(binding=3, dtype="u32", value=1024 * 1024),
             dict(binding=4, dtype="u32", value=1024),
             dict(binding=5, dtype="u32", value=8 * 1024)],
    flops=1024 * 1024,
    bytes=4 * (1024 * 1024 * 2 + 1024),
)

REGISTRY["fused_add_rms_norm"] = dict(
    metal_function="fused_add_rms_norm_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(1024, 1024), (1024, 1024)],
    output_shape=(1024, 1024),
    rtol=1e-3, atol=1e-3,
    grid=(1024, 1024, 1),
    scalars=[dict(binding=3, dtype="u32", value=1024),
             dict(binding=4, dtype="f32", value=1e-5)],
    flops=1024 * 1024 * 6,
    bytes=4 * 1024 * 1024 * 3,
)

REGISTRY["linear_bias"] = dict(
    metal_function="linear_bias_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1, 2),
    input_shapes=[(256, 256), (256, 256), (256,)],
    output_shape=(256, 256),
    rtol=1e-3, atol=1e-3,
    grid=(((256 * 256) + 255) // 256 * 256, 1, 1),
    scalars=[dict(binding=4, dtype="u32", value=256),
             dict(binding=5, dtype="u32", value=256),
             dict(binding=6, dtype="u32", value=256)],
    flops=2 * 256 * 256 * 256 + 256 * 256,
    bytes=4 * (256 * 256 * 3 + 256),
)

REGISTRY["bias_gelu"] = dict(
    metal_function="bias_gelu_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(1024, 1024), (1024,)],
    output_shape=(1024, 1024),
    rtol=1e-3, atol=1e-3,
    grid=(64 * 1024, 1, 1),
    scalars=[dict(binding=3, dtype="u32", value=1024 * 1024),
             dict(binding=4, dtype="u32", value=1024),
             dict(binding=5, dtype="u32", value=64 * 1024)],
    flops=1024 * 1024 * 6,
    bytes=4 * (1024 * 1024 * 2 + 1024),
)

REGISTRY["fused_qkv_projection"] = dict(
    metal_function="fused_qkv_projection_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(128, 512), (512, 192)],
    output_shape=(128, 192),
    rtol=1e-3, atol=1e-3,
    grid=(((128 * 192) + 255) // 256 * 256, 1, 1),
    scalars=[dict(binding=3, dtype="u32", value=128),
             dict(binding=4, dtype="u32", value=192),
             dict(binding=5, dtype="u32", value=512)],
    flops=2 * 128 * 192 * 512,
    bytes=4 * (128 * 512 + 512 * 192 + 128 * 192),
)

REGISTRY["attention_scores"] = dict(
    metal_function="attention_scores_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(128, 64), (128, 64)],
    output_shape=(128, 128),
    rtol=1e-3, atol=1e-3,
    grid=(1024, 128, 1),
    scalars=[dict(binding=3, dtype="u32", value=128),
             dict(binding=4, dtype="u32", value=64)],
    flops=128 * (128 * 64 * 2 + 128 * 5),
    bytes=4 * (128 * 64 * 2 + 128 * 128),
)

REGISTRY["nll_loss"] = dict(
    metal_function="nll_loss_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(1024, 1024), (1024, 1024)],
    output_shape=(1024,),
    rtol=1e-3, atol=1e-3,
    grid=(1024, 1024, 1),
    scalars=[dict(binding=3, dtype="u32", value=1024),
             dict(binding=4, dtype="u32", value=1024)],
    flops=1024 * 1024 * 2,
    bytes=4 * 1024 * 1024 * 2 + 1024 * 4,
)

REGISTRY["log_softmax_cross_entropy"] = dict(
    metal_function="log_softmax_cross_entropy_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(1024, 1024), (1024, 1024)],
    output_shape=(1024,),
    rtol=1e-3, atol=1e-3,
    grid=(1024, 1024, 1),
    scalars=[dict(binding=3, dtype="u32", value=1024),
             dict(binding=4, dtype="u32", value=1024)],
    flops=1024 * 1024 * 6,
    bytes=4 * 1024 * 1024 * 2 + 1024 * 4,
)

# scaled_dot_product: softmax(Q@K^T / sqrt(d)) @ V
REGISTRY["scaled_dot_product"] = dict(
    metal_function="scaled_dot_product_f32",
    threadgroup=(128, 1, 1),
    input_bindings=(0, 1, 2),
    input_shapes=[(128, 128), (128, 128), (128, 128)],
    output_shape=(128, 128),
    rtol=1e-3, atol=1e-3,
    grid=(128 * 128, 1, 1),
    scalars=[
        dict(binding=4, dtype="u32", value=128),
        dict(binding=5, dtype="u32", value=128),
    ],
    flops=128 * 128 * 128 * 4 + 128 * 128 * 5,
    bytes=128 * 128 * 4 * 4,
)

REGISTRY["llama_attention"] = dict(
    metal_function="llama_attention_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0, 1, 2),
    input_shapes=[(64, 128), (128, 128 + 2*2*32), (128, 128)],
    output_shape=(64, 128),
    rtol=1e-2, atol=1e-2,
    grid=(1024, 1, 1),
    scalars=[dict(binding=4, dtype="u32", value=64),
             dict(binding=5, dtype="u32", value=128),
             dict(binding=6, dtype="u32", value=4),
             dict(binding=7, dtype="u32", value=2),
             dict(binding=8, dtype="f32", value=10000.0)],
    flops=64 * (128 * (128+128) * 2 + 64 * 128 * 2 + 128 * 128 * 2),
    bytes=4 * (64*128*2 + 128*256 + 128*128),
)


# ===== batch-added new kernels =====
REGISTRY["silu_residual"] = dict(
    metal_function="silu_residual_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(4096,), (4096,)],
    output_shape=(4096,),
    rtol=1e-4, atol=1e-4,
    grid=(4096, 1, 1),
    scalars=[dict(binding=3, dtype="u32", value=4096)],
    flops=4096 * 5,
    bytes=4 * (4096 + 4096 + 4096),
)

REGISTRY["gelu_residual"] = dict(
    metal_function="gelu_residual_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(4096,), (4096,)],
    output_shape=(4096,),
    rtol=1e-4, atol=1e-4,
    grid=(4096, 1, 1),
    scalars=[dict(binding=3, dtype="u32", value=4096)],
    flops=4096 * 5,
    bytes=4 * (4096 + 4096 + 4096),
)

REGISTRY["add_silu"] = dict(
    metal_function="add_silu_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(4096,), (4096,)],
    output_shape=(4096,),
    rtol=1e-4, atol=1e-4,
    grid=(4096, 1, 1),
    scalars=[dict(binding=3, dtype="u32", value=4096)],
    flops=4096 * 5,
    bytes=4 * (4096 + 4096 + 4096),
)

REGISTRY["mul_silu"] = dict(
    metal_function="mul_silu_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(4096,), (4096,)],
    output_shape=(4096,),
    rtol=1e-4, atol=1e-4,
    grid=(4096, 1, 1),
    scalars=[dict(binding=3, dtype="u32", value=4096)],
    flops=4096 * 5,
    bytes=4 * (4096 + 4096 + 4096),
)

REGISTRY["add_gelu"] = dict(
    metal_function="add_gelu_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(4096,), (4096,)],
    output_shape=(4096,),
    rtol=1e-4, atol=1e-4,
    grid=(4096, 1, 1),
    scalars=[dict(binding=3, dtype="u32", value=4096)],
    flops=4096 * 5,
    bytes=4 * (4096 + 4096 + 4096),
)

REGISTRY["mul_gelu"] = dict(
    metal_function="mul_gelu_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(4096,), (4096,)],
    output_shape=(4096,),
    rtol=1e-4, atol=1e-4,
    grid=(4096, 1, 1),
    scalars=[dict(binding=3, dtype="u32", value=4096)],
    flops=4096 * 5,
    bytes=4 * (4096 + 4096 + 4096),
)

REGISTRY["add_relu"] = dict(
    metal_function="add_relu_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(4096,), (4096,)],
    output_shape=(4096,),
    rtol=1e-4, atol=1e-4,
    grid=(4096, 1, 1),
    scalars=[dict(binding=3, dtype="u32", value=4096)],
    flops=4096 * 5,
    bytes=4 * (4096 + 4096 + 4096),
)

REGISTRY["mul_relu"] = dict(
    metal_function="mul_relu_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(4096,), (4096,)],
    output_shape=(4096,),
    rtol=1e-4, atol=1e-4,
    grid=(4096, 1, 1),
    scalars=[dict(binding=3, dtype="u32", value=4096)],
    flops=4096 * 5,
    bytes=4 * (4096 + 4096 + 4096),
)

REGISTRY["add_swish"] = dict(
    metal_function="add_swish_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(4096,), (4096,)],
    output_shape=(4096,),
    rtol=1e-4, atol=1e-4,
    grid=(4096, 1, 1),
    scalars=[dict(binding=3, dtype="u32", value=4096)],
    flops=4096 * 5,
    bytes=4 * (4096 + 4096 + 4096),
)

REGISTRY["residual_tanh"] = dict(
    metal_function="residual_tanh_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1),
    input_shapes=[(4096,), (4096,)],
    output_shape=(4096,),
    rtol=1e-4, atol=1e-4,
    grid=(4096, 1, 1),
    scalars=[dict(binding=3, dtype="u32", value=4096)],
    flops=4096 * 5,
    bytes=4 * (4096 + 4096 + 4096),
)

