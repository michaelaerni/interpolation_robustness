import typing

import flax
import jax.numpy as jnp


# noinspection PyAttributeOutsideInit
class LinearModel(flax.linen.Module):
    num_out: int
    weight_init: typing.Callable[
        [flax.linen.linear.PRNGKey, flax.linen.linear.Shape, flax.linen.linear.Dtype], flax.linen.linear.Array
    ] = flax.linen.linear.default_kernel_init
    intercept_init: typing.Callable[
        [flax.linen.linear.PRNGKey, flax.linen.linear.Shape, flax.linen.linear.Dtype], flax.linen.linear.Array
    ] = flax.linen.linear.zeros
    use_intercept: bool = False

    def setup(self):
        self.dense = flax.linen.Dense(
            self.num_out,
            use_bias=self.use_intercept,
            kernel_init=self.weight_init,
            bias_init=self.intercept_init
        )

    def __call__(self, inputs: jnp.ndarray):
        return self.dense(inputs)
