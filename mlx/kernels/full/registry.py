"""Full Set kernel registry — end-to-end model forward passes."""

REGISTRY = {}


# alexnet-mini: 3 conv + 2 fc. Input (1, 32, 32, 3) → (1, 10).
REGISTRY["alexnet"] = dict(
    metal_function="alexnet_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0, 1, 2, 3, 4, 5),
    input_shapes=[
        (1, 32, 32, 3),       # x
        (32, 5, 5, 3),         # W_c1
        (64, 3, 3, 32),        # W_c2
        (128, 3, 3, 64),       # W_c3
        (512, 256),            # W_fc1   (4*4*128 = 2048 → use 2*2*128 = 512 after 3 pools)
        (256, 10),             # W_fc2
    ],
    output_shape=(1, 10),
    rtol=1e-2, atol=1e-2,
    grid=(1024, 1, 1),
    scalars=[],
    flops=1 * (32*5*5*3*28*28 + 64*3*3*32*12*12 + 128*3*3*64*4*4 + 512*256 + 256*10) * 2,
    bytes=4 * (1*32*32*3 + 32*5*5*3 + 64*3*3*32 + 128*3*3*64 + 512*256 + 256*10 + 10),
)

# resnet-mini: stem + 1 residual block + GAP + FC. Input (1, 32, 32, 3) → (1, 10).
REGISTRY["resnet"] = dict(
    metal_function="resnet_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0, 1, 2, 3, 4),
    input_shapes=[
        (1, 32, 32, 3),       # x
        (16, 3, 3, 3),         # W_stem
        (16, 3, 3, 16),        # W_a
        (16, 3, 3, 16),        # W_b
        (16, 10),              # W_fc
    ],
    output_shape=(1, 10),
    rtol=1e-2, atol=1e-2,
    grid=(1024, 1, 1),
    scalars=[],
    flops=1 * (16*3*3*3*32*32 + 16*3*3*16*32*32*2 + 16*10) * 2,
    bytes=4 * (1*32*32*3 + 16*3*3*3 + 16*3*3*16*2 + 16*10 + 10),
)

REGISTRY["transformer_block"] = dict(
    metal_function="transformer_block_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0, 1, 2, 3, 4),
    input_shapes=[(64, 128), (128, 3*128), (128, 128), (128, 256), (256, 128)],
    output_shape=(64, 128),
    rtol=1e-2, atol=1e-2,
    grid=(1024, 1, 1),
    scalars=[dict(binding=6, dtype="u32", value=64),
             dict(binding=7, dtype="u32", value=128),
             dict(binding=8, dtype="u32", value=4),
             dict(binding=9, dtype="u32", value=256),
             dict(binding=10, dtype="f32", value=1e-5)],
    flops=64 * (128 * 384 * 2 + 64 * 128 * 2 + 128 * 128 * 2 + 128 * 256 * 2 + 256 * 128 * 2),
    bytes=4 * (64*128*3 + 128*384 + 128*128 + 128*256 + 256*128),
)


# densenet-mini: stem + 2 dense layers (channel concat) + GAP + FC.
# Input (1,16,16,3) → (1,10).
REGISTRY["densenet"] = dict(
    metal_function="densenet_f32",
    threadgroup=(256, 1, 1),
    input_bindings=(0, 1, 2, 3, 4),
    input_shapes=[
        (1, 16, 16, 3),        # x
        (12, 3, 3, 3),         # W_stem
        (12, 3, 3, 12),        # W_d1
        (12, 3, 3, 24),        # W_d2
        (36, 10),              # W_fc
    ],
    output_shape=(1, 10),
    rtol=1e-2, atol=1e-2,
    grid=(256, 1, 1),
    scalars=[],
    flops=2 * (12*3*3*3*16*16 + 12*3*3*12*16*16 + 12*3*3*24*16*16 + 36*10),
    bytes=4 * (1*16*16*3 + 12*3*3*3 + 12*3*3*12 + 12*3*3*24 + 36*10 + 10),
)


REGISTRY["llama_decoder_layer"] = dict(
    metal_function="llama_decoder_layer_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0, 1, 2, 3, 4),
    input_shapes=[(64, 128), (128, 128 + 2*2*32), (128, 128), (128, 2*256), (256, 128)],
    output_shape=(64, 128),
    rtol=1e-2, atol=1e-2,
    grid=(1024, 1, 1),
    scalars=[dict(binding=6, dtype="u32", value=64),
             dict(binding=7, dtype="u32", value=128),
             dict(binding=8, dtype="u32", value=4),
             dict(binding=9, dtype="u32", value=2),
             dict(binding=10, dtype="u32", value=256),
             dict(binding=11, dtype="f32", value=10000.0),
             dict(binding=12, dtype="f32", value=1e-5)],
    flops=64 * (128 * 256 * 2 + 64 * 128 * 2 + 128 * 128 * 2 + 128 * 512 * 2 + 256 * 128 * 2),
    bytes=4 * (64*128*3 + 128*256 + 128*128 + 128*512 + 256*128),
)

