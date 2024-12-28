import os
import sys
import math
import time
import psutil
import argparse
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from jaxtyping import Array, Int, PyTree, Key
from rich import print

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import Network
from src.dataset import load_data
from src.train import loss_fn


def measure_runtime(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    opt_state: PyTree,
    x: Int[Array, "batch 1 mels frames"],
    y: Int[Array, "batch 1 mels frames"],
    keys: Key[Array, " batch"],
    jit: bool = True,
    num_runs: int = 5,
) -> float:
    """Measures the average runtime of the model's forward pass.

    Args:
        model (eqx.Module): The Network instance.
        optim (optax.GradientTransformation): Optimizer.
        opt_state (PyTree): Optimizer state.
        x (Int[Array, batch 1 mels frames]): Input batch.
        y (Int[Array, batch 1 mels frames]): Target batch.
        keys (Key[Array, batch]): Random keys for JAX.
        jit (bool, optional): Whether to use JAX's jit compilation. Defaults to True.
        num_runs (int, optional): Number of runs to average. Defaults to 5.

    Returns:
        float: Average runtime in seconds.
    """

    def singlepass(model, optim, opt_state, x, y, keys):
        """Forward + backward pass of the model."""
        loss, grads = loss_fn(model, x, y, keys)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return loss

    if jit:
        singlepass = eqx.filter_jit(singlepass)

    # Warm-up runs
    for _ in range(3):
        singlepass(model, optim, opt_state, x, y, keys)
    start = time.time()
    for _ in range(num_runs):
        loss = singlepass(model, optim, opt_state, x, y, keys)
        loss.block_until_ready()
    end = time.time()
    avg_runtime = (end - start) / num_runs
    return avg_runtime


def estimate_memory_usage(
    model: eqx.Module,
    batch_size: int,
    batch_shape: tuple[int, ...],
    dtype: str = "float32",
) -> float:
    """
    Estimates memory usage based on model parameters and input batch.
    Args:
        model (eqx.Module): The Network instance.
        batch_size (int): Size of the input batch.
        batch_shape (tuple[int, ...]): Shape of the input batch.
        dtype (str): Data type of the model parameters and inputs.
    Returns:
        float: Estimated memory usage in megabytes (MB).
    """
    bytes_per_param = jnp.dtype(dtype).itemsize
    # Total parameters
    total_params = sum(
        leaf.size for leaf in jax.tree.leaves(model) if eqx.is_array(leaf)
    )
    # Memory for parameters
    param_memory = total_params * bytes_per_param
    # Estimate memory for activations (forward + backward)
    activation_memory = (2 * param_memory * batch_size) + (
        1.5 * param_memory * batch_size
    )
    gradient_memory = param_memory
    # adamw optimizer. 32 bit counter variable
    optimizer_memory = 4 + (param_memory * 2)
    # data batch memory
    data_memory = math.prod(batch_shape) * bytes_per_param
    # Total memory in bytes. Add 10% safety margin for runtime overhead
    total_memory = (
        param_memory
        + activation_memory
        + gradient_memory
        + optimizer_memory
        + data_memory
    ) * 1.1

    # Convert to MB
    total_memory_mb = total_memory / (1024**2)
    return total_memory_mb


def get_memory_usage() -> float:
    """Get memory usage of current program."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024**2)


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer_dims",
        type=int,
        required=True,
        nargs="+",
        help="List of layer dimensions",
    )
    parser.add_argument("--kernel_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--dtype", type=str, default="float32", required=True)
    parser.add_argument("--batch_size", type=int, default=32, required=True)
    args = parser.parse_args(args)

    train_dataloader = load_data("data", "train", args.batch_size)
    val_dataloader = load_data("data", "val", args.batch_size)
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches in train dataloader: {len(train_dataloader):,}")
    print(f"Number of batches in val dataloader: {len(val_dataloader):,}")

    key = jr.key(21)
    model_key, train_key = jr.split(key)
    model = Network(args.layer_dims, args.kernel_size, model_key)

    optim = optax.adamw(learning_rate=0.0004, weight_decay=0.1)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    leaves, _ = jax.tree.flatten(model)
    num_params = sum([leaf.size for leaf in leaves if eqx.is_array(leaf)])
    print(f"Total number of model parameters: {num_params:,}")
    # model_str = eqx.tree_pformat(model)
    # print(model_str)
    # sys.exit()

    key, *sample_keys = jr.split(train_key, train_dataloader.batch_size + 1)
    sample_keys = jnp.array(sample_keys)
    x_sample, y_sample = next(iter(train_dataloader))
    print(f"Input shape: {x_sample.shape}")
    print(f"Target shape: {y_sample.shape}")

    print("\n[bold green]1. Runtime Benchmarking (forward+backward pass)[/bold green]")
    runtime_jit = measure_runtime(
        model, optim, opt_state, x_sample, y_sample, sample_keys, jit=True
    )
    print(f"Average runtime (JIT-compiled): {runtime_jit * 1e3:,.3f} ms")
    runtime_no_jit = measure_runtime(
        model, optim, opt_state, x_sample, y_sample, sample_keys, jit=False
    )
    print(f"Average runtime (Non-JIT): {runtime_no_jit * 1e3:,.3f} ms")
    speedup = runtime_no_jit / runtime_jit if runtime_jit > 0 else float("inf")
    print(f"Speedup (JIT / Non-JIT): {speedup:.2f}x")
    print(
        f"Estimated training time (JIT-compiled; {args.epochs} epochs): {runtime_jit * args.epochs / 60:.3f} minutes"
    )

    # 2. Memory Usage
    print("\n[bold green]2. Memory Usage Estimation[/bold green]")
    estimated_memory_mb = estimate_memory_usage(
        model, args.batch_size, x_sample.shape, dtype=args.dtype
    )
    estimated_memory_gb = estimated_memory_mb / 1024
    print(
        f"Estimated memory usage: {estimated_memory_mb:,.2f} MB / {estimated_memory_gb:,.2f} GB"
    )

    # run forward and backward pass then get actual memory usage
    loss_fn(model, x_sample, y_sample, sample_keys)
    memory_usage_mb = get_memory_usage()
    memory_usage_gb = memory_usage_mb / 1024
    print(
        f"Current memory usage: {memory_usage_mb:,.2f} MB / {memory_usage_gb:,.2f} GB"
    )

    # 3. FLOPs Estimation
    def forward(x, inference, key):
        return jax.vmap(model, in_axes=(0, None, 0))(x, inference, key)

    lowered = eqx.filter_jit(forward).lower(x_sample, False, sample_keys)
    compiled = lowered.compile().compiled
    flops = compiled.cost_analysis()[0]["flops"]
    gflops = flops / 1e9
    tflops = flops / 1e12
    tflops_training = tflops * args.epochs * len(train_dataloader) * 2
    print("\n[bold green]3. FLOPs Estimation[/bold green]")
    print(f"Estimated FLOPs per forward pass: {gflops:.3f} GFLOPs")
    print(
        f"Estimated total FLOPs for training: {flops:.1e} FLOPS ({tflops_training:.3f} TFLOPs)"
    )


if __name__ == "__main__":
    main()
