<p align="center">
  <img src="assets/logo.png" alt="MetalBench" width="240">
</p>

# MetalBench

Agentic harness for authoring + benchmarking Apple Metal GPU kernels against [MLX](https://github.com/ml-explore/mlx) reference implementations. Modeled on [KernelBench](https://github.com/ScalingIntelligence/KernelBench), swapping CUDA → Metal and PyTorch → MLX.



Hello, so working with SuperKittens, llm authored kernels + testing has been soomething i've itilized, and found to be something that inspired this offical repo. This contains harneseses and code used to benchmark against baseline mlx versions. Kernels don't differ much, just how threadgroups are utilized. One of the main differences was performance across a array of newer m chips. I soon realized this was a we-have-kernel-bench-at-home version I made, so I polished some stuff and releasing this as a benchmark + as a repo to continue your own metal kernel testing using any model. Much of this repo is orgaized/inspired by KernelBench, so props for them!


## Kernels
I've decided to split it into 4 types:

* Common Set - 100 of the most used operations & bits of a normal kernel, stuff such as matrix multipication in it's most basic shape, convolutions & layernorm to name a few
* Standard Set - 50 of most semi-difficult, common fused kernels, which brings more variations & use of memory and compute into play
* Full Set - 25 of most large scale, multiple operation kernels, similar to full steps, and can span entire model architectures

## Evaluation

So the actual benchmark measures 2 things, accuracy + performance. Accuracy is the error difference from mlx implementation to 

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
make            # builds host binary + every src/kernels/*.metal
```

Requires Xcode command line tools (`xcrun -sdk macosx metal`) and an Apple Silicon (or Metal-capable) Mac.

## Authoring a task

A task is a Python file that exports `task: Task`. The author provides:

- a `.metal` kernel (compiled to `build/<name>.metallib` by `make`)
- `make_inputs(seed)` — returns input `mx.array`s in binding order
- `outputs` — what the host should allocate and read back
- `scalars_fn(inputs)` — scalar bindings (sizes, alphas, etc.)
- `grid_fn(inputs)` + `threadgroup` — launch config for `dispatchThreads:`
- `reference(*inputs)` — MLX reference implementation

## Running

```bash
metalbench tasks/level1/<task>.py --iters 100
```

Output is a JSON dict with `correct`, `speedup` (MLX median / kernel median), and per-output max error.

## Manifest contract (host ↔ Python)

The Python harness writes a JSON manifest describing one launch; the C++ host consumes it. See the docstring at the top of `src/host/main.mm` for the schema.
