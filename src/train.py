import time
import logging
from rich.live import Live
from rich.panel import Panel
from rich.console import Group
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Float, Key, PyTree, Scalar
from torch.utils.data import DataLoader
from src.utils import configure_pbar, save_checkpoint, compute_metrics_from_logits

# Get the logger instance
logger = logging.getLogger("train")


@eqx.filter_value_and_grad(has_aux=True)
def forward_pass(
    model: eqx.Module,
    x: Float[Array, "batch 1 mels frames"],
    y: Float[Array, " batch"],
    state: eqx.nn.State,
    keys: Key[Array, " batch"],
) -> tuple[Scalar, tuple[Float[Array, "batch 1"], eqx.nn.State]]:
    """Forward pass of model and compute loss."""
    logits, state = jax.vmap(
        model, axis_name="batch", in_axes=(0, None, None, 0), out_axes=(0, None)
    )(x, False, state, keys)
    loss = optax.losses.sigmoid_binary_cross_entropy(logits, y).mean()
    return loss, (logits, state)


def validation_forward_pass(
    model: eqx.Module,
    x: Float[Array, "batch 1 mels frames"],
    y: Float[Array, " batch"],
    state: eqx.nn.State,
) -> tuple[Scalar, Float[Array, "batch 1"]]:
    logits, _ = jax.vmap(
        model, axis_name="batch", in_axes=(0, None, None, None), out_axes=(0, None)
    )(x, True, state, None)
    loss = optax.losses.sigmoid_binary_cross_entropy(logits, y).mean()
    return loss, logits


@eqx.filter_jit
@eqx.debug.assert_max_traces(max_traces=1)
def train_step(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    opt_state: PyTree,
    x: Float[Array, "batch 1 mels frames"],
    y: Float[Array, " batch"],
    state: eqx.nn.State,
    keys: Key[Array, " batch"],
) -> tuple[eqx.Module, PyTree, Scalar, Float[Array, "batch 1"], Scalar, eqx.nn.State]:
    """Single training step for a batch of data. Forward pass, compute loss/grads, update weights."""
    (loss, (logits, state)), grads = forward_pass(model, x, y, state, keys)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    grad_norm = optax.tree_utils.tree_l2_norm(grads)
    return model, opt_state, loss, logits, grad_norm, state


@eqx.filter_jit
def validate_step(
    inference_model: eqx.Module,
    x: Float[Array, "batch 1 mels frames"],
    y: Float[Array, " batch"],
    state: eqx.nn.State,
) -> tuple[Scalar, Float[Array, "batch 1"]]:
    loss, logits = validation_forward_pass(inference_model, x, y, state)
    return loss, logits


def train(
    model: eqx.Module,
    state: eqx.nn.State,
    optim: optax.GradientTransformation,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    key: Key[Array, ""],
    num_epochs: int,
    checkpoint_freq: int,
    checkpoint_name: str,
    dtype: str,
) -> tuple[eqx.Module, eqx.nn.State]:
    """Train the model."""
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    main_pbar, eval_pbar = configure_pbar()
    panel = Panel(
        Group(main_pbar, eval_pbar),
        title="Training Model",
        style="gold1",
    )

    with Live(panel):
        main_task = main_pbar.add_task("[red]Training model...", total=num_epochs)
        val_task = eval_pbar.add_task(
            "[magenta1]Evaluating model on validation set...",
            total=len(val_dataloader),
            visible=False,
        )
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            total_samples = 0
            total_tp = total_fp = total_fn = total_correct = 0
            for i, (x_batch, y_batch) in enumerate(train_dataloader):
                x_batch, y_batch = x_batch.astype(dtype), y_batch.astype(dtype)
                lr = opt_state.hyperparams["learning_rate"]
                start = time.time()
                # train phase
                key, *subkeys = jr.split(key, len(x_batch) + 1)
                subkeys = jnp.array(subkeys)
                model, opt_state, loss, logits, grad_norm, state = train_step(
                    model, optim, opt_state, x_batch, y_batch, state, subkeys
                )

                step_time = (time.time() - start) * 1e3
                loss = loss.item()
                grad_norm = grad_norm.item()
                lr = lr.item()
                # Calculate accuracy, precision, and recall
                tp, fp, fn, _, correct = compute_metrics_from_logits(logits, y_batch)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                epoch_loss += loss
                total_correct += correct
                total_samples += x_batch.shape[0]

                logger.info(
                    f"Step [{i + 1:04d}/{len(train_dataloader):04d}] | Train Loss: {loss:.4f} | lr: {lr:.6f} | Grad Norm: {grad_norm:.3f} | Step Time: {step_time:04.0f}ms",
                    extra={
                        "mode": "step",
                        "step": i + 1,
                        "epoch": epoch,
                        "step_train_loss": round(loss, 4),
                        "epoch_train_loss": round(epoch_loss, 4),
                        "learning_rate": round(lr, 6),
                        "grad_norm": round(grad_norm, 3),
                        "step_time": round(step_time, 4),
                    },
                )

            epoch_loss /= len(train_dataloader)
            epoch_accuracy = float(total_correct / total_samples) * 100.0
            epoch_precision = total_tp / (total_tp + total_fp + 1e-9)
            epoch_recall = total_tp / (total_tp + total_fn + 1e-9)

            main_pbar.update(main_task, advance=1)
            # validation phase
            eval_pbar.update(val_task, visible=True)
            val_loss = 0.0
            val_total_samples = 0
            val_tp = val_fp = val_fn = val_tn = val_correct = 0
            inference_model = eqx.nn.inference_mode(model)
            for x_val, y_val in val_dataloader:
                x_val, y_val = x_val.astype(dtype), y_val.astype(dtype)
                loss, logits = validate_step(inference_model, x_val, y_val, state)
                val_loss += loss
                tp, fp, fn, tn, correct = compute_metrics_from_logits(logits, y_val)
                val_tp += tp
                val_fp += fp
                val_fn += fn
                val_tn += tn
                val_correct += correct
                val_total_samples += x_val.shape[0]
                eval_pbar.update(val_task, advance=1)
            eval_pbar.reset(val_task, visible=False)

            # Average and store loss
            val_loss = (val_loss / len(val_dataloader)).item()
            val_accuracy = float(val_correct / val_total_samples) * 100.0
            val_precision = val_tp / (val_tp + val_fp + 1e-9)
            val_recall = val_tp / (val_tp + val_fn + 1e-9)

            logger.info(
                f"Epoch [{epoch:04d}/{num_epochs:04d}] "
                f"| Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Precision: {epoch_precision:.3f}, Recall: {epoch_recall:.3f} "
                f"| Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, Precision: {val_precision:.3f}, Recall: {val_recall:.3f}",
                extra={
                    "mode": "epoch",
                    "step": i + 1,
                    "epoch": epoch,
                    "epoch_train_loss": round(epoch_loss, 4),
                    "epoch_train_accuracy": round(epoch_accuracy, 4),
                    "epoch_train_precision": round(epoch_precision, 4),
                    "epoch_train_recall": round(epoch_recall, 4),
                    "epoch_val_loss": round(val_loss, 4),
                    "epoch_val_accuracy": round(val_accuracy, 4),
                    "epoch_val_precision": round(val_precision, 4),
                    "epoch_val_recall": round(val_recall, 4),
                    "epoch_val_tp": val_tp,
                    "epoch_val_fp": val_fp,
                    "epoch_val_tn": val_tn,
                    "epoch_val_fn": val_fn,
                },
            )

            if (epoch % checkpoint_freq) == 0:
                ckpt_name = f"{checkpoint_name}-{epoch:03d}-chkpt.eqx"
                logger.info(f"Checkpointing model to disk: {ckpt_name}")
                save_checkpoint(ckpt_name, model, state)
    return model, state
