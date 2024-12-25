import pytest
import numpy as np
import jax.random as jr

@pytest.fixture
def key():
    return jr.key(42)

@pytest.fixture
def mock_data_dir(tmp_path):
    """Create small fake .npy files for X_train / y_train, etc."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create dummy data for testing
    X_dummy = np.random.rand(10, 64, 64)  # (num_samples, mels, frames)
    y_dummy = np.random.randint(0, 2, size=(10,), dtype=np.int32)

    np.save(data_dir / "X_test.npy", X_dummy)
    np.save(data_dir / "y_test.npy", y_dummy)
    return data_dir