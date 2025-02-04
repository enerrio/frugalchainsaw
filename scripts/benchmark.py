import os
import sys
import math
import time
import psutil
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import pyrallis
import optax
from jaxtyping import Array, Float, PyTree, Key
from rich import print

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_train import TrainConfig
from src.model import Network, reinit_model_params
from src.dataset import load_data
from src.train import forward_pass
from src.utils import compute_fc_in_dim


def measure_runtime(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    opt_state: PyTree,
    x: Float[Array, "batch 1 mels frames"],
    y: Float[Array, " batch"],
    state: eqx.nn.State,
    keys: Key[Array, " batch"],
    jit: bool = True,
    num_runs: int = 5,
) -> float:
    """Measures the average runtime of the model's forward + backward pass.

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

    def singlepass(model, optim, opt_state, x, y, state, keys):
        """Forward + backward pass of the model."""
        (loss, _), grads = forward_pass(model, x, y, state, keys)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return loss

    if jit:
        singlepass = eqx.filter_jit(singlepass)

    # Warm-up runs
    for _ in range(3):
        singlepass(model, optim, opt_state, x, y, state, keys)
    start = time.time()
    for _ in range(num_runs):
        loss = singlepass(model, optim, opt_state, x, y, state, keys)
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
    # If arguments were provided, replace sys.argv so Pyrallis sees them
    if args is not None and len(args) > 0:
        sys.argv = [sys.argv[0]] + args

    cfg = pyrallis.parse(config_class=TrainConfig)

    data_dir = "data" if cfg.normalization_mode == "global" else "data_binwise"
    train_dataloader = load_data(data_dir, "train", cfg.batch_size)
    val_dataloader = load_data(data_dir, "val", cfg.batch_size)
    print(f"Batch size: {cfg.batch_size}")
    print(f"Number of batches in train dataloader: {len(train_dataloader):,}")
    print(f"Number of batches in val dataloader: {len(val_dataloader):,}")

    key = jr.key(21)
    fc_in_dim = compute_fc_in_dim(
        cfg.layer_dims,
        cfg.kernel_size,
        train_dataloader.dataset.features.shape[-2],
        train_dataloader.dataset.features.shape[-1],
    )
    model_key, train_key = jr.split(key)
    model, state = eqx.nn.make_with_state(Network)(
        cfg.layer_dims, fc_in_dim, cfg.fc_out_dim, cfg.kernel_size, cfg.dtype, model_key
    )
    model = reinit_model_params(model, cfg.dtype, model_key)

    optim = optax.adamw(learning_rate=0.0004, weight_decay=0.1)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    leaves, _ = jax.tree.flatten(model)
    num_params = sum([leaf.size for leaf in leaves if eqx.is_array(leaf)])
    print(f"Total number of model parameters: {num_params:,}")

    key, *sample_keys = jr.split(train_key, train_dataloader.batch_size + 1)
    sample_keys = jnp.array(sample_keys)
    x_sample, y_sample = next(iter(train_dataloader))
    x_sample, y_sample = x_sample.astype(cfg.dtype), y_sample.astype(cfg.dtype)
    print(f"Input shape: {x_sample.shape}")
    print(f"Target shape: {y_sample.shape}")

    print("\n[bold green]1. Runtime Benchmarking (forward+backward pass)[/bold green]")
    runtime_jit = measure_runtime(
        model, optim, opt_state, x_sample, y_sample, state, sample_keys, jit=True
    )
    print(f"Average runtime (JIT-compiled): {runtime_jit * 1e3:,.3f} ms")
    runtime_no_jit = measure_runtime(
        model, optim, opt_state, x_sample, y_sample, state, sample_keys, jit=False
    )
    print(f"Average runtime (Non-JIT): {runtime_no_jit * 1e3:,.3f} ms")
    speedup = runtime_no_jit / runtime_jit if runtime_jit > 0 else float("inf")
    print(f"Speedup (JIT / Non-JIT): {speedup:.2f}x")
    # Add an empirical overhead factor (typically 3-7x depending on setup)
    overhead_factor = 7.0  # For data loading, logging, checkpointing, etc.
    estimated_train_time = runtime_jit * len(train_dataloader) * cfg.epochs / 60
    estimated_train_time += runtime_jit * len(val_dataloader) * cfg.epochs / 60
    estimated_train_time *= overhead_factor
    print(
        f"Estimated training time (JIT-compiled; {cfg.epochs} epochs): {estimated_train_time:.3f} minutes"
        f" (includes {overhead_factor}x overhead factor)"
    )

    # 2. Memory Usage
    print("\n[bold green]2. Memory Usage Estimation[/bold green]")
    estimated_memory_mb = estimate_memory_usage(
        model, cfg.batch_size, x_sample.shape, dtype=cfg.dtype
    )
    estimated_memory_gb = estimated_memory_mb / 1024
    print(
        f"Estimated memory usage: {estimated_memory_mb:,.2f} MB / {estimated_memory_gb:,.2f} GB"
    )

    # run forward and backward pass then get actual memory usage
    forward_pass(model, x_sample, y_sample, state, sample_keys)
    memory_usage_mb = get_memory_usage()
    memory_usage_gb = memory_usage_mb / 1024
    print(
        f"Current memory usage: {memory_usage_mb:,.2f} MB / {memory_usage_gb:,.2f} GB"
    )

    # 3. FLOPs Estimation
    def forward(x, inference, state, key):
        return jax.vmap(
            model, axis_name="batch", in_axes=(0, None, None, 0), out_axes=(0, None)
        )(x, inference, state, key)

    lowered = eqx.filter_jit(forward).lower(x_sample, False, state, sample_keys)
    compiled = lowered.compile().compiled
    flops = compiled.cost_analysis()["flops"]
    gflops = flops / 1e9
    tflops = flops / 1e12
    tflops_training = tflops * cfg.epochs * len(train_dataloader) * 2
    print("\n[bold green]3. FLOPs Estimation[/bold green]")
    print(f"Estimated FLOPs per forward pass: {gflops:.3f} GFLOPs")
    print(
        f"Estimated total FLOPs for training: {flops:.1e} FLOPS ({tflops_training:.3f} TFLOPs)"
    )
