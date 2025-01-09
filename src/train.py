import time
import logging
from rich.live import Live
from rich.panel import Panel
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Float, Key, PyTree, Scalar
from torch.utils.data import DataLoader
from src.utils import configure_pbar, save_checkpoint

# Get the logger instance
logger = logging.getLogger("train")


@eqx.filter_value_and_grad(has_aux=True)
def forward_pass(
    model: eqx.Module,
    x: Float[Array, "batch 1 mels frames"],
    y: Float[Array, " batch"],
    keys: Key[Array, " batch"],
) -> tuple[Scalar, Float[Array, "batch 1"]]:
    """Forward pass of model and compute loss."""
    logits = jax.vmap(model, in_axes=(0, None, 0))(x, False, keys)
    loss = optax.losses.sigmoid_binary_cross_entropy(logits, y).mean()
    return loss, logits


def validation_forward_pass(
    model: eqx.Module,
    x: Float[Array, "batch 1 mels frames"],
    y: Float[Array, " batch"],
) -> tuple[Scalar, Float[Array, "batch 1"]]:
    logits = jax.vmap(model, in_axes=(0, None, None))(x, True, None)
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
    keys: Key[Array, " batch"],
) -> tuple[eqx.Module, PyTree, Scalar, Float[Array, "batch 1"], Scalar]:
    """Single training step for a batch of data. Forward pass, compute loss/grads, update weights."""
    (loss, logits), grads = forward_pass(model, x, y, keys)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    grad_norm = optax.tree_utils.tree_l2_norm(grads)
    return model, opt_state, loss, logits, grad_norm


@eqx.filter_jit
def validate_step(
    inference_model: eqx.Module,
    x: Float[Array, "batch 1 mels frames"],
    y: Float[Array, " batch"],
) -> tuple[Scalar, Float[Array, "batch 1"]]:
    loss, logits = validation_forward_pass(inference_model, x, y)
    return loss, logits


def compute_metrics_from_logits(
    logits: Float[Array, "batch 1"], labels: Float[Array, " batch"]
) -> tuple[int, int, int, int, int]:
    """Calculate TP, FP, FN, and correct predictions from logits and labels."""
    preds = jax.nn.sigmoid(logits)
    preds_bin = (preds >= 0.5).astype(jnp.float32)
    preds_bin = preds_bin.squeeze(axis=-1)
    y_bin = labels.astype(jnp.float32)

    tp = jnp.sum((preds_bin == 1) & (y_bin == 1)).item()
    fp = jnp.sum((preds_bin == 1) & (y_bin == 0)).item()
    fn = jnp.sum((preds_bin == 0) & (y_bin == 1)).item()
    tn = jnp.sum((preds_bin == 0) & (y_bin == 0)).item()
    correct = jnp.sum(preds_bin == y_bin).item()
    return tp, fp, fn, tn, correct


def train(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    key: Key[Array, ""],
    num_epochs: int,
    checkpoint_freq: int,
    checkpoint_name: str,
    dtype: str,
) -> eqx.Module:
    """Train the model."""
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

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
                model, opt_state, loss, logits, grad_norm = train_step(
                    model, optim, opt_state, x_batch, y_batch, subkeys
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
                    f"Step [{i+1:07d}/{len(train_dataloader):07d}] | Train Loss: {loss:.4f} | lr: {lr:.6f} | Grad Norm: {grad_norm:.3f} | Step Time: {step_time:04.0f}ms",
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
            main_pbar.update(val_task, visible=True)
            val_loss = 0.0
            val_total_samples = 0
            val_tp = val_fp = val_fn = val_tn = val_correct = 0
            inference_model = eqx.nn.inference_mode(model)
            for x_val, y_val in val_dataloader:
                x_val, y_val = x_val.astype(dtype), y_val.astype(dtype)
                loss, logits = validate_step(inference_model, x_val, y_val)
                val_loss += loss
                tp, fp, fn, tn, correct = compute_metrics_from_logits(logits, y_val)
                val_tp += tp
                val_fp += fp
                val_fn += fn
                val_tn += tn
                val_correct += correct
                val_total_samples += x_val.shape[0]
                main_pbar.update(val_task, advance=1)
            main_pbar.reset(val_task, visible=False)

            # Average and store loss
            val_loss = (val_loss / len(val_dataloader)).item()
            val_accuracy = float(val_correct / val_total_samples) * 100.0
            val_precision = val_tp / (val_tp + val_fp + 1e-9)
            val_recall = val_tp / (val_tp + val_fn + 1e-9)

            logger.info(
                f"Epoch [{epoch:05d}/{num_epochs:05d}] "
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
                save_checkpoint(ckpt_name, model)
    return model
