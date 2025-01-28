import sys
import time
from glob import glob
from rich.live import Live
from rich.panel import Panel
import numpy as np
import equinox as eqx
import pyrallis
from run_train import TrainConfig
from src.dataset import load_data
from src.train import validate_step
from src.utils import (
    load_checkpoint,
    configure_pbar,
    make_prediction,
    compute_fc_in_dim,
)


def main(args=None):
    # If arguments were provided, replace sys.argv so Pyrallis sees them
    if args is not None and len(args) > 0:
        sys.argv = [sys.argv[0]] + args

    cfg = pyrallis.parse(config_class=TrainConfig)

    data_dir = "data" if cfg.normalization_mode == "global" else "data_binwise"
    test_dataloader = load_data(data_dir, "test", cfg.batch_size)
    print(f"Batch size: {cfg.batch_size}")
    print(f"Number of batches in test dataloader: {len(test_dataloader)}")

    model_path = glob(f"{cfg.exp_dir}/*final.eqx")[0]
    print(f"Loading final model checkpoint from: {model_path}")
    fc_in_dim = compute_fc_in_dim(
        cfg.layer_dims,
        cfg.kernel_size,
        test_dataloader.dataset.features.shape[-2],
        test_dataloader.dataset.features.shape[-1],
    )
    model, state = load_checkpoint(
        model_path, cfg.layer_dims, fc_in_dim, cfg.fc_out_dim, cfg.kernel_size
    )
    # TODO: Convert to dtype??

    inference_model = eqx.nn.inference_mode(model)
    preds = []

    _, eval_pbar = configure_pbar()
    panel = Panel(
        eval_pbar,
        title="Running Inference",
        style="gold1",
    )

    print("Predicting...")
    start = time.time()
    with Live(panel):
        test_task = eval_pbar.add_task(
            "[green1]Making predictions on test set...",
            total=len(test_dataloader),
        )
        for x_test, y_test in test_dataloader:
            x_test, y_test = x_test.astype(cfg.dtype), y_test.astype(cfg.dtype)
            _, logits = validate_step(inference_model, x_test, y_test, state)
            preds_batch = make_prediction(logits, threshold=0.5)
            preds.append(preds_batch)
            eval_pbar.update(test_task, advance=1)

    preds = np.concatenate(preds)
    print(f"Total prediction time: {time.time() - start:.2f} seconds.")
    pred_filename = f"{cfg.exp_dir}/test_predictions_{cfg.exp_name}.npy"
    np.save(pred_filename, preds)
    print(f"Predictions saved to {pred_filename}")
