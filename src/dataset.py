import os
from glob import glob
import einops
import numpy as np
import numpy.typing as npt
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from torch.utils.data import DataLoader, Dataset


class ChainsawDataset(Dataset):
    def __init__(self, data_dir: str, split: str) -> None:
        super().__init__()
        features_fp = glob(os.path.join(data_dir, f"X_{split}.npy"))[0]
        labels_fp = glob(os.path.join(data_dir, f"y_{split}.npy"))[0]
        self.features = np.load(features_fp)
        self.labels = np.load(labels_fp)
        # Add channel dimension
        self.features = einops.rearrange(
            self.features, "num_samples mels frames -> num_samples 1 mels frames"
        )
        # Testing TODO: remove
        # self.features = self.features[:32*1]
        # self.labels = self.labels[:32*1]
        # print(jnp.bincount(self.labels))

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        return self.features[idx], self.labels[idx]


def collate_fn(
    batch: list[tuple[npt.ArrayLike, npt.ArrayLike]],
) -> tuple[Float[Array, "batch 1 mels frames"], Int[Array, " batch"]]:
    """Convert tensors to Jax arrays."""
    input_batch, target_batch = zip(*batch)
    input_array = jnp.array(input_batch)
    target_array = jnp.array(target_batch)
    return input_array, target_array


def create_dataloader(
    data_dir: str,
    split: str,
    batch_size: int = 16,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Instantiate our custom Dataset and dataloader."""
    dataset = ChainsawDataset(data_dir=data_dir, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return dataloader


def load_data(
    data_dir: str,
    split: str,
    batch_size: int,
) -> DataLoader:
    """Load data and create dataloaders."""
    dataloader = create_dataloader(
        data_dir,
        split,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    return dataloader
