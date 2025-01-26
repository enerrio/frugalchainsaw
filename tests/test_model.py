import jax
import jax.numpy as jnp
import equinox as eqx
from src.model import Network, normal_init, reinit_model_params
from src.utils import compute_fc_in_dim


def test_normal_init_shapes_and_values(key):
    """Check that normal_init returns the correct shape, dtype, and approximate distribution."""
    shape = (4, 4)
    mean_val = 0.0
    std_val = 0.02
    arr = normal_init(key, shape, dtype=jnp.float32, mean=mean_val, std=std_val)

    assert arr.shape == shape
    assert arr.dtype == jnp.float32
    # Check that the standard deviation is close to the expected range.
    # This is a loose check; distribution can vary with small sample sizes.
    approx_std = jnp.std(arr)
    assert (
        0.01 < approx_std < 0.03
    ), f"Expected std close to {std_val}, got {approx_std}"


def test_reinit_model_params_weight_and_bias(key):
    """Verify that reinit_model_params resets weights to normal(0, 0.02) and biases to 0."""
    layer_dims = [1, 8, 16]
    fc_out_dim = 16
    kernel_size = 3
    height = 32
    width = 32
    fc_in_dim = compute_fc_in_dim(layer_dims, kernel_size, height, width)
    model = Network(layer_dims, fc_in_dim, fc_out_dim, kernel_size, key)

    # Reinit with different dtype for demonstration
    new_dtype = "bfloat16"
    model_reinit = reinit_model_params(model, new_dtype, key)

    # Extract all conv/linear layers
    def is_learnable_layer(x):
        return isinstance(x, eqx.nn.Conv2d) or isinstance(x, eqx.nn.Linear)

    # original_weights = [
    #     x.weight for x in jax.tree.leaves(model) if is_learnable_layer(x)
    # ]
    reinit_weights = [
        x.weight for x in jax.tree.leaves(model_reinit) if is_learnable_layer(x)
    ]

    # original_biases = [
    #     x.bias for x in jax.tree.leaves(model) if is_learnable_layer(x) and x.bias is not None
    # ]
    reinit_biases = [
        x.bias
        for x in jax.tree.leaves(model_reinit)
        if is_learnable_layer(x) and x.bias is not None
    ]

    # Biases after reinit should be all zeros
    for rb in reinit_biases:
        assert jnp.allclose(rb, 0.0), "Expected biases to be zero after reinit."

    # Weights after reinit should differ from original in general
    # and conform to normal(0, 0.02). We'll do a quick std check on the new weights.
    for w in reinit_weights:
        # Loose check on standard deviation
        w_std = w.std()
        assert 0.01 < w_std < 0.03, f"Expected weight std ~0.02, got {w_std}"


def test_network_init(key):
    """Test that Network initializes without errors."""
    layer_dims = [1, 8, 16]
    kernel_size = 3
    fc_out_dim = 16
    height = 32
    width = 32
    fc_in_dim = compute_fc_in_dim(layer_dims, kernel_size, height, width)
    model = Network(layer_dims, fc_in_dim, fc_out_dim, kernel_size, key)

    assert len(model.layers) == 2  # For 3 dims => 2 convolution layers
    assert isinstance(model.out_layer, eqx.nn.Linear)


def test_network_forward_pass(key):
    """Test a forward pass on a mock input to ensure shape correctness."""
    layer_dims = [1, 8, 16]  # in_channels=1 -> out_channels=16 after conv layers
    kernel_size = 3
    fc_out_dim = 16
    height = 64
    width = 64
    fc_in_dim = compute_fc_in_dim(layer_dims, kernel_size, height, width)
    model, state = eqx.nn.make_with_state(Network)(layer_dims, fc_in_dim, fc_out_dim, kernel_size, key)

    # Suppose `channels=1` and 'mels=64' and 'frames=64'
    # model.__call__(...) expects shape (1, mels, frames)
    x = jnp.ones((1, height, width))  # Single sample

    # Forward pass
    output, new_state = model(x, inference=True, state=state)


    # Since out_layer has 1 output unit, we expect shape (1, ) or scalar.
    # It might end up as shape (64,) if vmap is used across mels.
    assert output.shape[-1] == 1, f"Expected output shape (..., 1), got {output.shape}"
    assert new_state is not None, "Expected batch norm state to be returned"


def test_network_multiple_samples(key):
    """Test passing multiple inputs via vmap or batching logic if relevant."""
    layer_dims = [1, 8, 16]
    kernel_size = 3
    fc_out_dim = 16
    height = 64
    width = 64
    fc_in_dim = compute_fc_in_dim(layer_dims, kernel_size, height, width)
    model, state = eqx.nn.make_with_state(Network)(layer_dims, fc_in_dim, fc_out_dim, kernel_size, key)

    # Simulate a batch of 5 samples, each with shape (1, 64, 64)
    batch_size = 5
    x_batch = jnp.ones((batch_size, 1, height, width))

    # use vmap on the model
    batched_model = jax.vmap(model, in_axes=(0, None, None, None))
    outputs, states = batched_model(x_batch, True, state, None)

    # Check that we have a per-sample output
    assert outputs.shape == (batch_size, 1), f"Expected output shape ({batch_size}, 1), got {outputs.shape}"


def test_network_grad(key):
    """Simple gradient check to ensure model is trainable."""
    layer_dims = [1, 8, 16]
    kernel_size = 3
    fc_out_dim = 16
    height = 64
    width = 64
    fc_in_dim = compute_fc_in_dim(layer_dims, kernel_size, height, width)
    model, state = eqx.nn.make_with_state(Network)(layer_dims, fc_in_dim, fc_out_dim, kernel_size, key)

    x = jnp.ones((1, height, width), dtype=jnp.float32)
    y = jnp.array([1.0])  # Suppose we want a scalar target

    def loss_fn(m, inputs, target):
        pred, _ = m(inputs, inference=True, state=state)
        return ((pred - target) ** 2).squeeze()  # MSE

    grads = eqx.filter_grad(loss_fn)(model, x, y)
    assert grads is not None
