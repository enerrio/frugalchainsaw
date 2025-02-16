import json
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
import equinox as eqx
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
    Column,
)
from jaxtyping import Array, Float, Int
import matplotlib.pyplot as plt
from src.model import Network


def compute_fc_in_dim(
    layer_dims: list[int], kernel_size: int, height: int, width: int
) -> int:
    """Calculate fc layer input dim based on conv block."""
    padding = kernel_size // 2
    for i in range(len(layer_dims)-1):
        stride = 1 if i == 0 else 2
        height = (height + 2 * padding - kernel_size) // stride + 1
        width = (width + 2 * padding - kernel_size) // stride + 1
    final_out_channels = layer_dims[-1]
    return height * width * final_out_channels


def make_prediction(
    logits: Float[Array, "batch 1"], threshold: float = 0.5
) -> Int[Array, " batch"]:
    """Make predictions for a given batch of data."""
    preds = jax.nn.sigmoid(logits)
    preds_bin = (preds >= threshold).astype(jnp.int32)
    preds_bin = preds_bin.squeeze(axis=-1)

    return preds_bin


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


def save_checkpoint(filename: str, model: eqx.Module, state: eqx.nn.State) -> None:
    """Save model to disk."""
    with open(filename, "wb") as f:
        eqx.tree_serialise_leaves(f, (model, state))


def load_checkpoint(
    filename: str,
    layer_dims: list[int],
    fc_in_dim: int,
    fc_out_dim: int,
    kernel_size: int,
    dtype: str,
) -> tuple[eqx.Module, eqx.nn.State]:
    """Load saved model."""
    skeleton, state = eqx.nn.make_with_state(Network)(
        layer_dims, fc_in_dim, fc_out_dim, kernel_size, dtype, jr.key(21)
    )
    with open(filename, "rb") as f:
        model, state = eqx.tree_deserialise_leaves(f, (skeleton, state))
    return model, state


def configure_pbar() -> tuple[Progress, Progress]:
    """Setup rich progress bar to monitor a training run."""
    main_pbar = Progress(
        TextColumn(
            "[progress.description]{task.description}", table_column=Column(ratio=1)
        ),
        TextColumn(
            "{task.completed:,} of [underline]{task.total:,}[/underline] epochs completed"
        ),
        TextColumn("•"),
        BarColumn(bar_width=None, table_column=Column(ratio=2)),
        TaskProgressColumn(text_format="[progress.percentage]{task.percentage:>3.1f}%"),
        TimeElapsedColumn(),
        expand=True,
    )
    eval_pbar = Progress(
        TextColumn(
            "[progress.description]{task.description}", table_column=Column(ratio=1)
        ),
        TextColumn(
            "{task.completed:,} of [underline]{task.total:,}[/underline] batches completed"
        ),
        TextColumn("•"),
        BarColumn(bar_width=None, table_column=Column(ratio=2)),
        TaskProgressColumn(text_format="[progress.percentage]{task.percentage:>3.1f}%"),
        TimeElapsedColumn(),
        expand=True,
    )
    return main_pbar, eval_pbar


def read_log_file(logfile: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads a JSONL log file and returns DataFrames with training and val metrics."""
    records = []
    with open(logfile, "r") as f:
        for line in f:
            record = json.loads(line)
            records.append(record)
    df = pd.DataFrame(records)

    # Separate between step-wise and epoch-wise metrics
    step_df = df[df["mode"] == "step"].sort_values("step").reset_index(drop=True)
    epoch_df = df[df["mode"] == "epoch"].sort_values("step").reset_index(drop=True)
    return step_df, epoch_df


def plot_confusion_matrix(cm: np.ndarray, cm_name: str) -> None:
    """Plots a confusion matrix."""
    fig_cm, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(12, 5))

    im1 = ax1.imshow(cm, interpolation="nearest", cmap="coolwarm")
    ax1.set_title("Confusion Matrix")
    ax1.set_ylabel("True label")
    ax1.set_xlabel("Predicted label")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    plt.colorbar(im1, ax=ax1)
    # Add annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", color="black")

    # Normalize confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Heatmap 2: Normalized Confusion Matrix
    im2 = ax2.imshow(cm_norm, interpolation="nearest", cmap="coolwarm")
    ax2.set_title("Normalized Confusion Matrix")
    ax2.set_ylabel("True label")
    ax2.set_xlabel("Predicted label")
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    plt.colorbar(im2, ax=ax2)

    # Add annotations
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax2.text(
                j, i, f"{cm_norm[i, j]:.3f}", ha="center", va="center", color="black"
            )

    plt.savefig(cm_name)
    plt.close(fig_cm)


def plot_stats(logfile: str, plot_name: str, cm_name: str) -> None:
    """Plots training stats."""
    step_df, epoch_df = read_log_file(logfile)

    # Compute total training time from step timestamps
    step_df["timestamp"] = pd.to_datetime(step_df["timestamp"])
    total_training_time = str(step_df["timestamp"].max() - step_df["timestamp"].min()).split('.')[0]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # Plot loss
    axes[0, 0].plot(epoch_df["epoch_train_loss"], label="Training Loss")
    axes[0, 0].plot(epoch_df["epoch_val_loss"], label="Validation Loss")
    axes[0, 0].set_title("Model Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(step_df["step_time"], label="Training Step Time")
    axes[0, 1].set_title("Training Step Time")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Time (ms)")
    axes[0, 1].legend()

    # Plot accuracy
    axes[1, 0].plot(epoch_df["epoch_train_accuracy"], label="Training Accuracy")
    axes[1, 0].plot(epoch_df["epoch_val_accuracy"], label="Validation Accuracy")
    axes[1, 0].set_title("Model Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].legend()

    # Plot precision
    axes[1, 1].plot(
        epoch_df["epoch_train_precision"],
        label="Training Precision",
        linestyle="-",
        color="blue",
    )
    axes[1, 1].plot(
        epoch_df["epoch_val_precision"],
        label="Validation Precision",
        linestyle="--",
        color="blue",
    )
    axes[1, 1].set_title("Model Precision")

    # Plot recall
    axes[1, 1].plot(
        epoch_df["epoch_train_recall"],
        label="Training Recall",
        linestyle="-",
        color="orange",
    )
    axes[1, 1].plot(
        epoch_df["epoch_val_recall"],
        label="Validation Recall",
        linestyle="--",
        color="orange",
    )
    axes[1, 1].set_title("Model Recall + Precision")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f"Total Training Time: {total_training_time}", fontsize=16)
    plt.savefig(plot_name)
    plt.close(fig)

    # Create confusion matrix
    cm = np.array(
        [
            [
                epoch_df.iloc[-1, epoch_df.columns.get_loc("epoch_val_tp")],
                epoch_df.iloc[-1, epoch_df.columns.get_loc("epoch_val_tn")],
            ],
            [
                epoch_df.iloc[-1, epoch_df.columns.get_loc("epoch_val_fp")],
                epoch_df.iloc[-1, epoch_df.columns.get_loc("epoch_val_fn")],
            ],
        ],
        dtype=int,
    )
    plot_confusion_matrix(cm, cm_name)
