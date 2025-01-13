import os
import sys
import time
from pathlib import Path
from dataclasses import field
from functools import partial

from dataclasses import dataclass
import jax
import jax.random as jr
import optax
import equinox as eqx
import torch
import pyrallis

from src.dataset import load_data
from src.train import train
from src.logger import setup_logger
from src.model import Network, reinit_model_params
from src.utils import save_checkpoint


@partial(optax.inject_hyperparams, static_args="weight_decay")
def create_optimizer(learning_rate, weight_decay):
    return optax.chain(
        # optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.add_decayed_weights(weight_decay=weight_decay),
        optax.scale_by_learning_rate(learning_rate=learning_rate),
    )


@dataclass
class TrainConfig:
    """Training config."""

    seed: int = 42
    # Number of epochs to train for
    epochs: int = 10
    # Number of data samples in batch
    batch_size: int = 32
    # Model layer sizes
    layer_dims: list[int] = field(default_factory=lambda: [1, 32, 64])
    # Convolutional kernel size
    kernel_size: int = 3
    # Max learning rate
    learning_rate: float = 4e-4
    # Optimizer weight decay
    weight_decay: float = 0.1
    # Fraction of total training step to warm up learning rate to learning_rate
    warmup_percentage: float = 0.05
    # Fraction of max learning rate to decay to after warmup; 0 means decay to zero, 1 means no decay
    decay_percentage: float = 0.1
    # Data type for model weights
    dtype: str = "float32"
    # How often to save the model to disk
    checkpoint_freq: int = 5
    # Directory to store experiment results
    results_dir: Path = Path("results")
    # Experiment name
    exp_name: Path = Path("default_exp")

    @property
    def exp_dir(self) -> Path:
        return self.results_dir / self.exp_name


def main(args=None):
    # If arguments were provided, replace sys.argv so Pyrallis sees them
    if args is not None and len(args) > 0:
        sys.argv = [sys.argv[0]] + args

    cfg = pyrallis.parse(config_class=TrainConfig)
    os.makedirs(cfg.exp_dir, exist_ok=True)
    logfile = os.path.join(cfg.exp_dir, f"train_log_{cfg.exp_name}.jsonl")
    logger = setup_logger(log_file=logfile)
    logger.info(f"Logging training info to {logfile}")

    ckpt_dir = os.path.join(cfg.exp_dir, f"model-{cfg.exp_name}")
    logger.info(f"Saving model checkpoints to {cfg.exp_dir}")

    train_dataloader = load_data("data", "train", cfg.batch_size)
    val_dataloader = load_data("data", "val", cfg.batch_size)
    torch.manual_seed(cfg.seed)
    logger.info(f"Batch size: {cfg.batch_size}")
    logger.info(f"Number of batches in train dataloader: {len(train_dataloader)}")
    logger.info(f"Number of batches in val dataloader: {len(val_dataloader)}")

    key = jr.key(cfg.seed)
    logger.info(f"Creating model with dype: {cfg.dtype}")
    model_key, train_key = jr.split(key)
    model, state = eqx.nn.make_with_state(Network)(cfg.layer_dims, cfg.kernel_size, model_key)
    model = reinit_model_params(model, cfg.dtype, model_key)
    # model_str = eqx.tree_pformat(model)
    # print(model_str)

    # Calculate initial loss
    # import jax.numpy as jnp
    # initial_loss = -jnp.log(1.0 / 2.0)
    # logger.info(f"Initial loss should be around: {initial_loss:.3f}")
    # key, *sample_keys = jr.split(train_key, train_dataloader.batch_size + 1)
    # sample_keys = jnp.array(sample_keys)
    # x_sample, y_sample = next(iter(train_dataloader))
    # x_sample, y_sample = x_sample.astype(cfg.dtype), y_sample.astype(cfg.dtype)
    # logits = jax.vmap(model, in_axes=(0, None, 0))(x_sample, False, sample_keys)
    # loss = optax.losses.sigmoid_binary_cross_entropy(
    #     logits, y_sample
    # ).mean().item()
    # logits_mean = jnp.mean(logits).item()
    # logits_std = jnp.std(logits).item()
    # logger.info(
    #     f"Logits mean: {logits_mean:.3f}, std: {logits_std:.3f}, dtype: {logits.dtype}"
    # )
    # probs = jax.nn.sigmoid(logits)
    # logger.info("Initial prediction stats:")
    # logger.info(f"Mean: {jnp.mean(probs):.3f}")
    # logger.info(f"Std: {jnp.std(probs):.3f}")
    # logger.info(f"Min: {jnp.min(probs):.3f}")
    # logger.info(f"Max: {jnp.max(probs):.3f}")
    # logger.info(f"Actual initial loss is: {loss:.3f}")
    # logger.info(f"> conv2d weight dtype: {model.layers[0].weight.dtype}")
    # logger.info(f"> conv2d bias dtype: {model.layers[0].bias.dtype}")
    # logger.info(f"> conv2d weight mean/std: {model.layers[0].weight.mean(), model.layers[0].weight.std()}")
    # logger.info(f"> conv2d bias mean/std: {model.layers[0].bias.mean(), model.layers[0].bias.std()}")
    # logger.info(f"> conv2d weight dtype: {model.layers[-1].weight.dtype}")
    # logger.info(f"> conv2d bias dtype: {model.layers[-1].bias.dtype}")
    # logger.info(f"> out_layer dtype: {model.out_layer.weight.dtype}")
    # logger.info(f"> out_layer weight mean/std: {model.out_layer.weight.mean(), model.out_layer.weight.std()}")
    # logger.info(f"> out_layer bias mean/std: {model.out_layer.bias.mean(), model.out_layer.bias.std()}")
    # sys.exit()

    end_value = cfg.learning_rate * cfg.decay_percentage
    total_steps = cfg.epochs * len(train_dataloader)
    warmup_steps = int(total_steps * cfg.warmup_percentage)
    logger.info(f"Total training steps: {total_steps:,}")
    logger.info(f"Number of warmup steps: {warmup_steps}")
    lr_scheduler = optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=end_value,
    )
    optim = create_optimizer(lr_scheduler, cfg.weight_decay)

    leaves, _ = jax.tree.flatten(model)
    num_params = sum([leaf.size for leaf in leaves if eqx.is_array(leaf)])
    logger.info(f"Total number of model parameters: {num_params:,}")

    logger.info("Training...")
    start = time.time()
    model, state = train(
        model=model,
        state=state,
        optim=optim,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        key=train_key,
        num_epochs=cfg.epochs,
        checkpoint_freq=cfg.checkpoint_freq,
        checkpoint_name=ckpt_dir,
        dtype=cfg.dtype,
    )
    logger.info(f"Total training time: {(time.time()-start) / 60:.2f} minutes.")
    logger.info("Complete!")
    save_checkpoint(f"{ckpt_dir}-final.eqx", model, state)
    logger.info(f"Final model saved to disk: {ckpt_dir}-final.eqx")
