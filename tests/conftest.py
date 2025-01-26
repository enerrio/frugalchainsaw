import pytest
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr


class DummyModel(eqx.Module):
    """Simple model for testing purposes."""

    # For example, we can store just one parameter: a bias term
    bias: jnp.ndarray

    def __call__(self, x, inference, state, key):
        # Simple linear forward pass for demonstration
        # shape of x: (batch, 1, mels, frames)
        return x.mean() + self.bias, state


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


@pytest.fixture
def dummy_data(key):
    """Generate some dummy inputs (x) and labels (y)."""
    x = jr.normal(key, (4, 1, 16, 16), dtype=jnp.float32)
    y = jnp.array([0, 1, 1, 0], dtype=jnp.int32)
    return x, y


@pytest.fixture
def dummy_model():
    """Create a simple eqx model."""
    return DummyModel(bias=jnp.array(0.1))
