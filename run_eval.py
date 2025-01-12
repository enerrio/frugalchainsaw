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
from src.train import validate_step, compute_metrics_from_logits
from src.utils import load_checkpoint, configure_pbar, plot_confusion_matrix


def main(args=None):
    # If arguments were provided, replace sys.argv so Pyrallis sees them
    if args is not None and len(args) > 0:
        sys.argv = [sys.argv[0]] + args

    cfg = pyrallis.parse(config_class=TrainConfig)

    test_dataloader = load_data("data", "test", cfg.batch_size)
    print(f"Batch size: {cfg.batch_size}")
    print(f"Number of batches in test dataloader: {len(test_dataloader)}")

    model_path = glob(f"{cfg.exp_dir}/*final.eqx")[0]
    print(f"Loading final model checkpoint from: {model_path}")
    model = load_checkpoint(model_path, cfg.layer_dims, cfg.kernel_size)
    # TODO: Convert to dtype??

    test_loss = 0.0
    test_total_samples = 0
    test_tp = test_fp = test_fn = test_tn = test_correct = 0
    inference_model = eqx.nn.inference_mode(model)

    _, eval_pbar = configure_pbar()
    panel = Panel(
        eval_pbar,
        title="Evaluating Model",
        style="gold1",
    )

    print("Evaluating...")
    start = time.time()
    with Live(panel):
        test_task = eval_pbar.add_task(
            "[green1]Evaluating model on test set...",
            total=len(test_dataloader),
        )
        for x_test, y_test in test_dataloader:
            x_test, y_test = x_test.astype(cfg.dtype), y_test.astype(cfg.dtype)
            loss, logits = validate_step(inference_model, x_test, y_test)
            test_loss += loss
            tp, fp, fn, tn, correct = compute_metrics_from_logits(logits, y_test)
            test_tp += tp
            test_fp += fp
            test_fn += fn
            test_tn += tn
            test_correct += correct
            test_total_samples += x_test.shape[0]
            eval_pbar.update(test_task, advance=1)

    # Average and store loss
    test_loss = (test_loss / len(test_dataloader)).item()
    test_accuracy = float(test_correct / test_total_samples) * 100.0
    test_precision = test_tp / (test_tp + test_fp + 1e-9)
    test_recall = test_tp / (test_tp + test_fn + 1e-9)

    print(f"Total evaluation time: {time.time() - start:.2f} seconds.")
    print(
        f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%, Precision: {test_precision:.3f}, Recall: {test_recall:.3f}",
    )
    # Plot confusion matrix
    cm = np.array(
        [
            [
                test_tp,
                test_tn,
            ],
            [
                test_fp,
                test_fn,
            ],
        ],
        dtype=int,
    )
    cm_name = f"{cfg.exp_dir}/test_cm_{cfg.exp_name}.png"
    plot_confusion_matrix(cm, cm_name)
    print(f"Confusion matrix saved to {cm_name}")
