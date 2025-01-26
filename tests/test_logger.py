import json
import logging
from src.logger import setup_logger


def test_setup_logger_creates_logger_with_expected_handlers(tmp_path):
    log_file = tmp_path / "test.log"
    logger = setup_logger(name="train_test", log_file=str(log_file))

    # Check logger has two handlers: Rich console and file
    assert len(logger.handlers) == 2

    # Check the file handler uses our custom JsonFormatter
    file_handler = next(
        h for h in logger.handlers if isinstance(h, logging.FileHandler)
    )
    assert file_handler.formatter.__class__.__name__ == "JsonFormatter"


def test_json_formatter_outputs_valid_json(tmp_path):
    log_file = tmp_path / "test.log"
    logger = setup_logger(name="train_test_json", log_file=str(log_file))

    logger.info("Testing JSON format", extra={"mode": "train", "step": 10})
    with open(log_file, "r") as f:
        line = f.readline().strip()
        data = json.loads(line)
        # Check some keys in the JSON
        assert data["mode"] == "train"
        assert data["step"] == 10
        assert "timestamp" in data


def test_step_filter_allows_records_with_mode(tmp_path):
    log_file = tmp_path / "test.log"
    logger = setup_logger(name="train_test_filter", log_file=str(log_file))

    # This record has `mode`, so it should pass
    logger.info("Allowed message", extra={"mode": "train"})
    # This one has no `mode`, so StepFilter should block it from file logs
    logger.info("Blocked message")

    with open(log_file, "r") as f:
        lines = f.readlines()

    assert len(lines) == 1
    assert json.loads(lines[0])["mode"] == "train"


def test_log_record_includes_additional_fields(tmp_path):
    log_file = tmp_path / "test.log"
    logger = setup_logger(name="train_test_fields", log_file=str(log_file))

    # Log a record with various extra fields
    logger.info(
        "Logging additional fields",
        extra={
            "mode": "eval",
            "epoch": 5,
            "epoch_val_accuracy": 0.88,
            "grad_norm": 1.23,
        },
    )
    with open(log_file, "r") as f:
        data = json.loads(f.readline().strip())

    assert data["mode"] == "eval"
    assert data["epoch"] == 5
    assert data["epoch_val_accuracy"] == 0.88
    assert data["grad_norm"] == 1.23
