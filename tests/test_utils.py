import pytest
from src.utils import compute_fc_in_dim


@pytest.mark.parametrize(
    "layer_dims,kernel_size,height,width,fcdim_true",
    [
        ([1, 32, 64], 3, 128, 71, 147456),
        ([1, 16], 3, 64, 64, 65536),
        ([1, 16, 32], 3, 128, 128, 131072),
        ([3, 16, 16], 5, 32, 32, 4096),
        ([1, 32, 64], 3, 100, 80, 128000),
    ],
)
def test_compute_fc_in_dim(layer_dims, kernel_size, height, width, fcdim_true):
    fcdim = compute_fc_in_dim(layer_dims, kernel_size, height, width)
    assert fcdim == fcdim_true
