import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray


def normal_init(
    key: PRNGKeyArray,
    shape: tuple[int, int],
    dtype: str,
    mean: float = 0.0,
    std: float = 0.02,
) -> Array:
    return mean + std * jr.normal(key, shape=shape, dtype=dtype)


def reinit_model_params(model: eqx.Module, dtype: str, key: PRNGKeyArray) -> eqx.Module:
    """Custom weight initialization."""
    # Split keys for each category
    key, key_linear = jr.split(key)

    # Linear layer weights: normal(0,0.02)
    is_learnable_layer = lambda x: isinstance(x, eqx.nn.Conv2d) or isinstance(
        x, eqx.nn.Linear
    )
    linear_layers = [
        x
        for x in jax.tree.leaves(model, is_leaf=is_learnable_layer)
        if is_learnable_layer(x)
    ]
    w_shapes = [layer.weight.shape for layer in linear_layers]
    b_shapes = [layer.bias.shape for layer in linear_layers if layer.bias is not None]
    w_keys = jr.split(key_linear, len(w_shapes))
    new_weights = [
        normal_init(k, s, dtype, 0.0, 0.02) for k, s in zip(w_keys, w_shapes)
    ]
    # bias should be 0
    new_biases = [jnp.zeros(s, dtype=dtype) for s in b_shapes]

    # Replace weights
    model = eqx.tree_at(
        lambda m: [
            x.weight
            for x in jax.tree.leaves(m, is_leaf=is_learnable_layer)
            if is_learnable_layer(x)
        ],
        model,
        new_weights,
    )
    # Replace biases
    model = eqx.tree_at(
        lambda m: [
            x.bias
            for x in jax.tree.leaves(m, is_leaf=is_learnable_layer)
            if is_learnable_layer(x) and x.bias is not None
        ],
        model,
        new_biases,
    )
    return model


class Network(eqx.Module):
    layers: list
    out_layer: eqx.nn.Linear

    def __init__(self, layer_dims: list[int], kernel_size: int, key: PRNGKeyArray):
        keys = jr.split(key, len(layer_dims))
        self.layers = []
        for i, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            self.layers.append(
                eqx.nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, key=keys[i])
            )
        self.out_layer = eqx.nn.Linear(layer_dims[-1], 1, key=keys[-1])

    def __call__(
        self,
        x: Float[Array, "1 mels frames"],
        inference: bool = False,
        key: PRNGKeyArray = None,
    ) -> Float[Array, " prediction"]:
        for layer in self.layers:
            x = layer(x)
            # batchnorm
            x = jax.nn.relu(x)
        # Global average pooling to compress (channels, height, width) -> (channels,)
        x = jnp.mean(x, axis=(-2, -1))
        x = self.out_layer(x)
        return x
