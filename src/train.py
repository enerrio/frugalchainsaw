import time
import logging
from rich.live import Live
from rich.panel import Panel
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Int, Key, PyTree, Scalar
from torch.utils.data import DataLoader
from src.utils import configure_pbar, save_checkpoint

# Get the logger instance
logger = logging.getLogger("train")


@eqx.filter_value_and_grad
def loss_fn(
    model: eqx.Module,
    x: Int[Array, "batch 1 mels frames"],
    y: Int[Array, "batch 1 mels frames"],
    keys: Key[Array, " batch"],
) -> Scalar:
    """Forward pass of model and compute loss."""
    logits = jax.vmap(model, in_axes=(0, None, 0))(x, False, keys)
    loss = optax.losses.sigmoid_binary_cross_entropy(logits, y)
    return loss.mean()


@eqx.filter_jit
@eqx.debug.assert_max_traces(max_traces=1)
def train_step(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    opt_state: PyTree,
    x: Int[Array, "batch 1 mels frames"],
    y: Int[Array, "batch 1 mels frames"],
    keys: Key[Array, " batch"],
) -> tuple[eqx.Module, PyTree, Scalar, Scalar]:
    """Single training step for a batch of data. Forward pass, compute loss/grads, update weights."""
    loss, grads = loss_fn(model, x, y, keys)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    grad_norm = optax.tree_utils.tree_l2_norm(grads)
    return model, opt_state, loss, grad_norm


def validation_loss_fn(
    model: eqx.Module,
    x: Int[Array, "batch 1 mels frames"],
    y: Int[Array, "batch 1 mels frames"],
) -> Scalar:
    logits = jax.vmap(model, in_axes=(0, None, None))(x, True, None)
    loss = optax.sigmoid_binary_cross_entropy(logits, y)
    return loss.mean()


@eqx.filter_jit
def validate_step(
    inference_model: eqx.Module,
    x: Int[Array, "batch 1 mels frames"],
    y: Int[Array, "batch 1 mels frames"],
) -> Scalar:
    loss = validation_loss_fn(inference_model, x, y)
    return loss


def train(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    key: Key[Array, ""],
    num_epochs: int,
    checkpoint_freq: int,
    checkpoint_name: str,
    dtype: str
) -> eqx.Module:
    """Train the model."""
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    tokens_seen = 0
    main_pbar = configure_pbar()
    panel = Panel(
        main_pbar,
        title="Training Model",
        style="gold1",
    )

    # with progress:
    with Live(panel):
        main_task = main_pbar.add_task("[red]Training model...", total=num_epochs)
        val_task = main_pbar.add_task(
            "[magenta1]Evaluating model on validation set...",
            total=len(val_dataloader),
            visible=False,
        )
        for epoch in range(1, num_epochs+1):
            for i, (x_batch, y_batch) in enumerate(train_dataloader):
                x_batch, y_batch = x_batch.astype(dtype), y_batch.astype(dtype)
                lr = opt_state.hyperparams["learning_rate"]
                start = time.time()
                # train phase
                key, *subkeys = jr.split(key, len(x_batch) + 1)
                subkeys = jnp.array(subkeys)
                model, opt_state, loss, grad_norm = train_step(
                    model, optim, opt_state, x_batch, y_batch, subkeys
                )

                step_time = (time.time() - start) * 1e3
                loss = loss.item()
                grad_norm = grad_norm.item()
                lr = lr.item()

                logger.info(
                    f"Step [{i+1:07d}/{len(train_dataloader):07d}] | Train Loss: {loss:.4f} | lr: {lr:.6f} | Grad Norm: {grad_norm:.3f} | Step Time: {step_time:04.0f}ms",
                    extra={
                        "mode": "training",
                        "step": i+1,
                        "train_loss": round(loss, 4),
                        "val_loss": None,
                        "learning_rate": round(lr, 6),
                        "grad_norm": round(grad_norm, 3),
                        "step_time": round(step_time, 4),
                    },
                )

            main_pbar.update(main_task, advance=1)
            # validation phase
            main_pbar.update(val_task, visible=True)
            val_loss = 0.0
            inference_model = eqx.nn.inference_mode(model)
            for x_val, y_val in val_dataloader:
                x_val, y_val = x_val.astype(dtype), y_val.astype(dtype)
                val_loss += validate_step(inference_model, x_val, y_val)
                main_pbar.update(val_task, advance=1)
            main_pbar.reset(val_task, visible=False)

            # Average and store loss
            val_loss /= len(val_dataloader)
            val_loss = val_loss.item()

            logger.info(
                f"Step [{i+1:07d}/{len(train_dataloader):07d}] | Val Loss: {val_loss:.4f}",
                extra={
                    "mode": "validation",
                    "step": i+1,
                    "train_loss": None,
                    "val_loss": round(val_loss, 4),
                    "learning_rate": None,
                    "grad_norm": None,
                    "step_time": None,
                },
            )

            if (epoch % checkpoint_freq) == 0:
                ckpt_name = f"{checkpoint_name}-{epoch:03d}-chkpt.eqx"
                logger.info(f"Checkpointing model to disk: {ckpt_name}")
                save_checkpoint(ckpt_name, model)
    return model
