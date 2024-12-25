import numpy as np
import jax.numpy as jnp
from torch.utils.data import DataLoader
from src.dataset import ChainsawDataset, collate_fn, create_dataloader, load_data


def test_chainsaw_dataset_init_length(mock_data_dir):
    """Test that ChainsawDataset is initialized properly and length is correct."""
    dataset = ChainsawDataset(data_dir=str(mock_data_dir), split="test")
    assert len(dataset) == 10


def test_chainsaw_dataset_get_item(mock_data_dir):
    """Test that __getitem__ returns the correct shape and type."""
    dataset = ChainsawDataset(data_dir=str(mock_data_dir), split="test")
    features, label = dataset[0]

    # Expect shape to match X_dummy’s shape minus first dimension
    assert features.shape == (1, 64, 64)
    assert isinstance(features, np.ndarray)
    assert isinstance(label, np.int32)


def test_collate_fn():
    """Test that collate_fn returns JAX arrays with the correct shapes."""
    batch = [
        (np.random.rand(64, 64), np.array(0)),
        (np.random.rand(64, 64), np.array(1)),
    ]
    inputs, targets = collate_fn(batch)

    assert isinstance(inputs, jnp.ndarray)
    assert isinstance(targets, jnp.ndarray)
    assert inputs.shape == (2, 64, 64)
    assert targets.shape == (2,)


def test_create_dataloader(mock_data_dir):
    """Test that create_dataloader returns a DataLoader with correct batch size."""
    dataloader = create_dataloader(
        data_dir=str(mock_data_dir), split="test", batch_size=4
    )
    assert isinstance(dataloader, DataLoader)

    batch = next(iter(dataloader))
    inputs, targets = batch
    # Check batch shape
    assert inputs.shape == (4, 1, 64, 64)
    assert targets.shape == (4,)


def test_load_data(mock_data_dir):
    """Test that load_data returns a DataLoader."""
    dataloader = load_data(
        data_dir=str(mock_data_dir),
        split="test",
        batch_size=4,
    )
    assert isinstance(dataloader, DataLoader)
    batch = next(iter(dataloader))
    inputs, targets = batch
    assert inputs.shape == (4, 1, 64, 64)
    assert targets.shape == (4,)
