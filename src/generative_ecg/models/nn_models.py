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

# class MLP(linen.Module):
#     features: Sequence[int]
#     activation: linen.Module = linen.softplus

#     @linen.compact
#     def __call__(self, x):
#         x = x.ravel()
#         for feat in self.features[:-1]:
#             x = self.activation(linen.Dense(feat)(x))
#         x = linen.Dense(self.features[-1])(x)

#         return x
    
    
# class CNN(linen.Module):
#     output_dim: int
#     activation: linen.Module = linen.relu
    
#     @linen.compact
#     def __call__(self, x):
#         x = jax.numpy.transpose(x, (1, 0)) # to (batch_size, time, channel)
#         x = linen.Conv(features=12, kernel_size=(10,))(x)
#         x = self.activation(x)
#         x = linen.avg_pool(x, window_shape=(2,), strides=(2,))
#         x = linen.Conv(features=12, kernel_size=(10,))(x)
#         x = self.activation(x)
#         x = linen.avg_pool(x, window_shape=(2,), strides=(2,))
#         x = x.ravel()
#         x = linen.Dense(features=128)(x)
#         x = self.activation(x)
#         x = linen.Dense(features=self.output_dim)(x).ravel()
        
#         return x


# class NCSN(linen.Module):
#     activation: linen.Module = linen.elu
#     num_features: int = 64
#     normalizer: linen.Module = layer_utils.ConditionalInstanceNorm2dPlus
#     get_sigmas: Callable = get_sigmas
#     interpolation: str = 'bilinear'

#     @linen.compact
#     def __call__(self, x, labels, train=True):
#         sigmas = self.get_sigmas()
        
#         # h = 2 * x - 1. # [0, 1] -> [-1, 1]
#         h = x
#         h = layer_utils.ncsn_conv(h, self.num_features, stride=1, bias=True,
#                              kernel_size=3)
#         # print(f"labels: {labels}")
        
#         # ResNet backbone
#         h = layer_utils.ConditionalResidualBlock(
#             self.num_features, resample=None, activation=self.activation, 
#             normalizer=self.normalizer
#         )(h, labels)
#         layer1 = layer_utils.ConditionalResidualBlock(
#             self.num_features, resample=None, activation=self.activation, 
#             normalizer=self.normalizer
#         )(h, labels)
        
#         h = layer_utils.ConditionalResidualBlock(
#             2 * self.num_features, resample='down', activation=self.activation,
#             normalizer=self.normalizer
#         )(layer1, labels)
#         layer2 = layer_utils.ConditionalResidualBlock(
#             2 * self.num_features, resample=None, activation=self.activation, 
#             normalizer=self.normalizer
#         )(h, labels)
        
#         h = layer_utils.ConditionalResidualBlock(
#              2 * self.num_features, resample='down', activation=self.activation,
#             normalizer=self.normalizer, dilation=2
#         )(layer2, labels)
#         layer3 = layer_utils.ConditionalResidualBlock(
#             2 * self.num_features, resample=None, activation=self.activation,
#             normalizer=self.normalizer, dilation=2
#         )(h, labels)
        
#         h = layer_utils.ConditionalResidualBlock(
#             2 * self.num_features, resample='down', activation=self.activation,
#             normalizer=self.normalizer, dilation=4
#         )(layer3, labels)
#         layer4 = layer_utils.ConditionalResidualBlock(
#             2 * self.num_features, resample=None, activation=self.activation,
#             normalizer=self.normalizer, dilation=4
#         )(h, labels)
        
#         # U-Net with RefineBlocks
#         ref1 = layer_utils.CondRefineBlock(
#             layer4.shape[1:2], 2 * self.num_features,
#             normalizer=self.normalizer, activation=self.activation,
#             interpolation=self.interpolation, start=True
#         )([layer4], labels)
#         ref2 = layer_utils.CondRefineBlock(
#             layer3.shape[1:2], 2 * self.num_features,
#             normalizer=self.normalizer, activation=self.activation,
#             interpolation=self.interpolation,
#         )([layer3, ref1], labels)
#         ref3 = layer_utils.CondRefineBlock(
#             layer2.shape[1:2], 2 * self.num_features,
#             normalizer=self.normalizer, activation=self.activation,
#             interpolation=self.interpolation,
#         )([layer2, ref2], labels)
#         ref4 = layer_utils.CondRefineBlock(
#             layer1.shape[1:2], self.num_features,
#             normalizer=self.normalizer, activation=self.activation,
#             interpolation=self.interpolation, end=True
#         )([layer1, ref3], labels)
#         h = self.normalizer()(ref4, labels)
#         h = self.activation(h)
#         h = layer_utils.ncsn_conv(h, x.shape[-1], kernel_size=3)

#         used_sigmas = sigmas[labels].reshape(
#             (x.shape[0], *([1] * len(x.shape[1:]))))
        
#         return h / used_sigmas

# # Encoder that returns Gaussian moments
# class Encoder(linen.Module):
#     features: Sequence[int]
#     activation: linen.Module = linen.relu

#     @linen.compact
#     def __call__(self, x):
#         x = x.ravel()
#         for feat in self.features[:-1]:
#             x = self.activation(linen.Dense(feat)(x))
#         y1 = linen.Dense(self.features[-1])(x)
#         y2 = linen.Dense(self.features[-1])(x)
#         y2 = linen.softplus(y2)

#         return y1, y2


# # CNN-based encoder that returns Gaussian moments
# class CNNEncoder(linen.Module):
#     output_dim: int
#     activation: linen.Module = linen.relu

#     @linen.compact
#     def __call__(self, x):
#         x = jax.numpy.transpose(x, (1, 0)) # to (batch_size, time, channel)
#         x = linen.Conv(features=12, kernel_size=(10,))(x)
#         x = self.activation(x)
#         x = linen.avg_pool(x, window_shape=(2,), strides=(2,))
#         x = linen.Conv(features=12, kernel_size=(10,))(x)
#         x = self.activation(x)
#         x = linen.avg_pool(x, window_shape=(2,), strides=(2,))
#         x = x.ravel()
#         x = linen.Dense(features=128)(x)
#         x = self.activation(x)
#         y1 = linen.Dense(features=self.output_dim)(x).ravel()
#         y2 = linen.Dense(features=self.output_dim)(x).ravel()
#         y2 = linen.softplus(y2)

#         return y1, y2

# # Decoder
# class Decoder(linen.Module):
#     features: Sequence[int]
#     activation: linen.Module = linen.relu
#     use_bias: bool = True

#     @linen.compact
#     def __call__(self, x):
#         x = x.ravel()
#         for feat in self.features[:-1]:
#             x = self.activation(linen.Dense(feat)(x))
#         x = linen.Dense(self.features[-1], 
#                      use_bias=self.use_bias)(x)

#         return x