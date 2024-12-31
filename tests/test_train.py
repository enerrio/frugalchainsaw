import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
from src.train import forward_pass, validation_forward_pass, train_step


def test_forward_pass(dummy_model, dummy_data, key):
    """Test that forward_pass runs without error and returns a scalar."""
    x, y = dummy_data
    keys = jr.split(key, x.shape[0] + 1)[1:]
    (loss, _), _ = forward_pass(dummy_model, x, y, jnp.array(keys))
    # Expect a single scalar for the loss
    assert jnp.ndim(loss) == 0, "forward_pass should return a scalar value."


def test_validation_forward_pass(dummy_model, dummy_data):
    """Test that validation_forward_pass returns a scalar."""
    x, y = dummy_data
    loss, _ = validation_forward_pass(dummy_model, x, y)
    assert jnp.ndim(loss) == 0, "validation_forward_pass should return a scalar value."


def test_train_step(dummy_model, dummy_data, key):
    """
    Test train_step to ensure it updates the model parameters
    and returns the correct tuple structure.
    """
    x, y = dummy_data
    # Basic optimizer
    optim = optax.sgd(learning_rate=0.01)
    opt_state = optim.init(eqx.filter(dummy_model, eqx.is_array))

    # Split the keys for each example
    key, *subkeys = jr.split(key, len(x) + 1)
    subkeys = jnp.array(subkeys)

    new_model, _, loss, _, grad_norm = train_step(
        dummy_model, optim, opt_state, x, y, subkeys
    )

    # Check returned shapes/types
    assert isinstance(new_model, eqx.Module), "train_step should return an eqx.Module."
    assert loss.shape == (), "train_step should return a scalar loss."
    assert grad_norm.shape == (), "train_step should return a scalar grad_norm."

    # Ensure model parameters have changed (assuming a real gradient update)
    # If the parameter didn't change, the difference should be zero
    param_before = dummy_model.bias
    param_after = new_model.bias
    # There's a chance an extremely small step won't show up in float32,
    # so just ensure they're not identical
    assert not jnp.allclose(
        param_before, param_after
    ), "Model parameter(s) should update."
