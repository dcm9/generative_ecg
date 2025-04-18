from typing import Callable, Sequence
from . import layer_utils 
from .math_utils import get_sigmas

from functools import partial
from typing import Any, Optional, Sequence

import flax.linen
import jax
import jax.numpy

class ECGConv(flax.linen.Module):
    output_dim: int
    activation: flax.linen.Module = flax.linen.relu

    @flax.linen.compact
    def __call__(self, x):
        x = jax.numpy.transpose(x, (1, 0)) # to (batch_size, time, channel)
        x = flax.linen.Conv(features=12, kernel_size=(10,))(x)
        x = self.activation(x)
        x = flax.linen.avg_pool(x, window_shape=(2,), strides=(2,))
        x = flax.linen.Conv(features=12, kernel_size=(10,))(x)
        x = self.activation(x)
        x = flax.linen.avg_pool(x, window_shape=(2,), strides=(2,))
        x = x.ravel()
        x = flax.linen.Dense(features=128)(x)
        x = self.activation(x)
        x = flax.linen.Dense(features=self.output_dim)(x).ravel()
        
        return x

# class MLP(flax.linen.Module):
#     features: Sequence[int]
#     activation: flax.linen.Module = flax.linen.softplus

#     @flax.linen.compact
#     def __call__(self, x):
#         x = x.ravel()
#         for feat in self.features[:-1]:
#             x = self.activation(flax.linen.Dense(feat)(x))
#         x = flax.linen.Dense(self.features[-1])(x)

#         return x
    
    
# class CNN(flax.linen.Module):
#     output_dim: int
#     activation: flax.linen.Module = flax.linen.relu
    
#     @flax.linen.compact
#     def __call__(self, x):
#         x = jax.numpy.transpose(x, (1, 0)) # to (batch_size, time, channel)
#         x = flax.linen.Conv(features=12, kernel_size=(10,))(x)
#         x = self.activation(x)
#         x = flax.linen.avg_pool(x, window_shape=(2,), strides=(2,))
#         x = flax.linen.Conv(features=12, kernel_size=(10,))(x)
#         x = self.activation(x)
#         x = flax.linen.avg_pool(x, window_shape=(2,), strides=(2,))
#         x = x.ravel()
#         x = flax.linen.Dense(features=128)(x)
#         x = self.activation(x)
#         x = flax.linen.Dense(features=self.output_dim)(x).ravel()
        
#         return x


# Encoder that returns Gaussian moments
class Encoder(flax.linen.Module):
    features: Sequence[int]
    activation: flax.linen.Module = flax.linen.relu

    @flax.linen.compact
    def __call__(self, x):
        x = x.ravel()
        for feat in self.features[:-1]:
            x = self.activation(flax.linen.Dense(feat)(x))
        y1 = flax.linen.Dense(self.features[-1])(x)
        y2 = flax.linen.Dense(self.features[-1])(x)
        y2 = flax.linen.softplus(y2)

        return y1, y2


# CNN-based encoder that returns Gaussian moments
class CNNEncoder(flax.linen.Module):
    output_dim: int
    activation: flax.linen.Module = flax.linen.relu

    @flax.linen.compact
    def __call__(self, x):
        x = jax.numpy.transpose(x, (1, 0)) # to (batch_size, time, channel)
        x = flax.linen.Conv(features=12, kernel_size=(10,))(x)
        x = self.activation(x)
        x = flax.linen.avg_pool(x, window_shape=(2,), strides=(2,))
        x = flax.linen.Conv(features=12, kernel_size=(10,))(x)
        x = self.activation(x)
        x = flax.linen.avg_pool(x, window_shape=(2,), strides=(2,))
        x = x.ravel()
        x = flax.linen.Dense(features=128)(x)
        x = self.activation(x)
        y1 = flax.linen.Dense(features=self.output_dim)(x).ravel()
        y2 = flax.linen.Dense(features=self.output_dim)(x).ravel()
        y2 = flax.linen.softplus(y2)

        return y1, y2

# Decoder
class Decoder(flax.linen.Module):
    features: Sequence[int]
    activation: flax.linen.Module = flax.linen.relu
    use_bias: bool = True

    @flax.linen.compact
    def __call__(self, x):
        x = x.ravel()
        for feat in self.features[:-1]:
            x = self.activation(flax.linen.Dense(feat)(x))
        x = flax.linen.Dense(self.features[-1], 
                     use_bias=self.use_bias)(x)

        return x