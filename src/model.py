import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray


def normal_init(
    key: PRNGKeyArray,
    shape: tuple[int, ...],
    dtype: str,
    mean: float = 0.0,
    std: float = 0.02,
) -> Array:
    return mean + std * jr.normal(key, shape=shape, dtype=dtype)


def reinit_model_params(model: eqx.Module, dtype: str, key: PRNGKeyArray) -> eqx.Module:
    """Custom weight initialization."""
    # Split keys for each category
    key, key_conv, key_out = jr.split(key, 3)
    # Increase std for conv layers based on input size
    conv_std = 0.2
    out_std = 0.2

    # Conv layer weights
    is_conv_layer = lambda x: isinstance(x, eqx.nn.Conv2d)
    conv_layers = [
        x for x in jax.tree.leaves(model, is_leaf=is_conv_layer) if is_conv_layer(x)
    ]
    w_shapes = [layer.weight.shape for layer in conv_layers]
    b_shapes = [layer.bias.shape for layer in conv_layers if layer.bias is not None]
    w_keys = jr.split(key_conv, len(w_shapes))
    new_weights = [
        normal_init(k, s, dtype, 0.0, conv_std) for k, s in zip(w_keys, w_shapes)
    ]
    new_biases = [jnp.full(s, 0.01, dtype=dtype) for s in b_shapes]
    # Initialize output layer separately
    out_w_shape = model.out_layer.weight.shape
    out_b_shape = model.out_layer.bias.shape
    new_out_weight = normal_init(key_out, out_w_shape, dtype, 0.0, out_std)
    new_out_bias = jnp.zeros(out_b_shape, dtype=dtype)
    # new_out_bias = jnp.full(out_b_shape, -1.0, dtype=dtype)

    # Replace conv weights
    model = eqx.tree_at(
        lambda m: [
            x.weight
            for x in jax.tree.leaves(m, is_leaf=is_conv_layer)
            if is_conv_layer(x)
        ],
        model,
        new_weights,
    )
    # Replace conv biases
    model = eqx.tree_at(
        lambda m: [
            x.bias
            for x in jax.tree.leaves(m, is_leaf=is_conv_layer)
            if is_conv_layer(x) and x.bias is not None
        ],
        model,
        new_biases,
    )
    # Replace output layer params
    model = eqx.tree_at(lambda m: m.out_layer.weight, model, new_out_weight)
    model = eqx.tree_at(lambda m: m.out_layer.bias, model, new_out_bias)
    return model


class Network(eqx.Module):
    layers: list
    bn_layers: list
    fc_layer: eqx.nn.Linear
    out_layer: eqx.nn.Linear

    def __init__(self, layer_dims: list[int], fc_dim: int, kernel_size: int, key: PRNGKeyArray):
        keys = jr.split(key, len(layer_dims) + 1)
        self.layers = []
        self.bn_layers = []
        for i, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            self.layers.append(
                eqx.nn.Conv2d(
                    in_dim,
                    out_dim,
                    kernel_size=kernel_size,
                    stride=(1 if i == 0 else 2),
                    padding=kernel_size // 2,
                    key=keys[i],
                )
            )
            self.bn_layers.append(eqx.nn.BatchNorm(out_dim, axis_name="batch"))
        self.fc_layer = eqx.nn.Linear(147456, fc_dim, key=keys[-2])
        self.out_layer = eqx.nn.Linear(fc_dim, 1, key=keys[-1])

    def __call__(
        self,
        x: Float[Array, "1 mels frames"],
        inference: bool = False,
        state: eqx.nn.State = None,
        key: PRNGKeyArray = None,
    ) -> tuple[Float[Array, " 1"], eqx.nn.State]:
        for layer, bn_layer in zip(self.layers, self.bn_layers):
            x = layer(x)
            x, state = bn_layer(x, state=state, inference=inference)
            x = jax.nn.leaky_relu(x)
        x = x.flatten()
        x = self.fc_layer(x)
        x = jax.nn.leaky_relu(x)
        x = self.out_layer(x)
        return x, state
