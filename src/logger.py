import logging
import json
from datetime import datetime, timezone
from rich.logging import RichHandler


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for log records."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "mode": record.mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": getattr(record, "step", None),
            "epoch": getattr(record, "epoch", None),
            "step_train_loss": getattr(record, "step_train_loss", None),
            "epoch_train_loss": getattr(record, "epoch_train_loss", None),
            "epoch_train_accuracy": getattr(record, "epoch_train_accuracy", None),
            "epoch_train_precision": getattr(record, "epoch_train_precision", None),
            "epoch_train_recall": getattr(record, "epoch_train_recall", None),
            "epoch_val_loss": getattr(record, "epoch_val_loss", None),
            "epoch_val_accuracy": getattr(record, "epoch_val_accuracy", None),
            "epoch_val_precision": getattr(record, "epoch_val_precision", None),
            "epoch_val_recall": getattr(record, "epoch_val_recall", None),
            "val_loss": getattr(record, "val_loss", None),
            "epoch_val_tp": getattr(record, "epoch_val_tp", None),
            "epoch_val_fp": getattr(record, "epoch_val_fp", None),
            "epoch_val_tn": getattr(record, "epoch_val_tn", None),
            "epoch_val_fn": getattr(record, "epoch_val_fn", None),
            # "test_loss": getattr(record, "test_loss", None),
            "learning_rate": getattr(record, "learning_rate", None),
            "grad_norm": getattr(record, "grad_norm", None),
            "step_time": getattr(record, "step_time", None),
        }
        return json.dumps(log_record)


class StepFilter(logging.Filter):
    """Custom filter to allow only log records that start with 'Step '."""

    def filter(self, record: logging.LogRecord) -> bool:
        return hasattr(record, "mode")


def setup_logger(
    name: str = "train", log_file: str = "training.log", level: int = logging.INFO
) -> logging.Logger:
    """Sets up a logger that outputs to the console using Rich and writes specific logs to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    # Console handler with Rich
    rich_handler = RichHandler(rich_tracebacks=True)
    rich_handler.setLevel(logging.INFO)
    logger.addHandler(rich_handler)

    # File handler for logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(StepFilter())

    json_formatter = JsonFormatter()
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger
