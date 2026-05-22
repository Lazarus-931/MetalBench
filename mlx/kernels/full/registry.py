"""Full Set kernel registry — end-to-end model forward passes."""

REGISTRY = {}

# mlp: 2-hidden-layer MLP. Input (16, 128), 3 weights, output (16, 10).
REGISTRY["mlp"] = dict(
    metal_function="mlp_f32",
    threadgroup=(1024, 1, 1),
    input_bindings=(0, 1, 2, 3),
    input_shapes=[(16, 128), (128, 512), (512, 128), (128, 10)],
    output_shape=(16, 10),
    rtol=1e-2, atol=1e-2,
    grid=(1024, 1, 1),
    scalars=[dict(binding=5, dtype="u32", value=16),
             dict(binding=6, dtype="u32", value=128),
             dict(binding=7, dtype="u32", value=512),
             dict(binding=8, dtype="u32", value=10)],
    flops=16 * (128 * 512 * 2 + 512 * 128 * 2 + 128 * 10 * 2),
    bytes=4 * (16 * 128 + 128 * 512 + 512 * 128 + 128 * 10 + 16 * 10),
)

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
