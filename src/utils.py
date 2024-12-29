import json
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
from src.model import Network


def save_checkpoint(filename: str, model: eqx.Module) -> None:
    """Save model to disk."""
    with open(filename, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def load_checkpoint(
    filename: str, layer_dims: list[int], kernel_size: int
) -> eqx.Module:
    """Load saved model."""
    skeleton = Network(layer_dims, kernel_size, jr.key(21))
    with open(filename, "rb") as f:
        return eqx.tree_deserialise_leaves(f, skeleton)


def configure_pbar() -> Progress:
    """Setup rich progress bar for monitoring training."""
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
    return main_pbar


def read_log_file(logfile: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads a JSONL log file and returns DataFrames with training and val metrics."""
    records = []
    with open(logfile, "r") as f:
        for line in f:
            record = json.loads(line)
            records.append(record)
    df = pd.DataFrame(records)

    # Separate training and validation
    train_df = df[df["mode"] == "training"].sort_values("step").reset_index(drop=True)
    val_df = df[df["mode"] == "validation"].sort_values("step").reset_index(drop=True)
    return train_df, val_df


import matplotlib.pyplot as plt


def plot_stats(logfile: str, plot_name: str) -> None:
    """Plots training stats."""
    train_df, val_df = read_log_file(logfile)
    # temp
    train_df = train_df.loc[train_df["step"] == 1]
    val_df = val_df.loc[val_df["step"] == 1]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot loss
    axes[0, 0].plot(train_df["loss"], label="Training Loss")
    axes[0, 0].plot(val_df["val_loss"], label="Validation Loss")
    axes[0, 0].set_title("Model Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    # Plot accuracy
    axes[0, 1].plot(train_df["accuracy"], label="Training Accuracy")
    axes[0, 1].plot(val_df["val_accuracy"], label="Validation Accuracy")
    axes[0, 1].set_title("Model Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()

    # Plot precision
    axes[1, 0].plot(train_df["precision"], label="Training Precision")
    axes[1, 0].plot(val_df["val_precision"], label="Validation Precision")
    axes[1, 0].set_title("Model Precision")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].legend()

    # Plot recall
    axes[1, 1].plot(train_df["recall"], label="Training Recall")
    axes[1, 1].plot(val_df["val_recall"], label="Validation Recall")
    axes[1, 1].set_title("Model Recall")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(plot_name)
