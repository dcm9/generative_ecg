from typing import Callable, Sequence

from flax import linen as nn
import jax.nn.initializers as init
import jax.numpy


from . import layer_utils as layers
from .math_utils import get_sigmas

from functools import partial
from typing import Any, Optional, Sequence


class MLP(nn.Module):
    features: Sequence[int]
    activation: nn.Module = nn.softplus

    @nn.compact
    def __call__(self, x):
        x = x.ravel()
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)

        return x
    
    
class CNN(nn.Module):
    output_dim: int
    activation: nn.Module = nn.relu
    
    @nn.compact
    def __call__(self, x):
        x = jax.numpy.transpose(x, (1, 0)) # to (batch_size, time, channel)
        x = nn.Conv(features=12, kernel_size=(10,))(x)
        x = self.activation(x)
        x = nn.avg_pool(x, window_shape=(2,), strides=(2,))
        x = nn.Conv(features=12, kernel_size=(10,))(x)
        x = self.activation(x)
        x = nn.avg_pool(x, window_shape=(2,), strides=(2,))
        x = x.ravel()
        x = nn.Dense(features=128)(x)
        x = self.activation(x)
        x = nn.Dense(features=self.output_dim)(x).ravel()
        
        return x


class NCSN(nn.Module):
    activation: nn.Module = nn.elu
    num_features: int = 64
    normalizer: nn.Module = layers.ConditionalInstanceNorm2dPlus
    get_sigmas: Callable = get_sigmas
    interpolation: str = 'bilinear'

    @nn.compact
    def __call__(self, x, labels, train=True):
        sigmas = self.get_sigmas()
        
        # h = 2 * x - 1. # [0, 1] -> [-1, 1]
        h = x
        h = layers.ncsn_conv(h, self.num_features, stride=1, bias=True,
                             kernel_size=3)
        # print(f"labels: {labels}")
        
        # ResNet backbone
        h = layers.ConditionalResidualBlock(
            self.num_features, resample=None, activation=self.activation, 
            normalizer=self.normalizer
        )(h, labels)
        layer1 = layers.ConditionalResidualBlock(
            self.num_features, resample=None, activation=self.activation, 
            normalizer=self.normalizer
        )(h, labels)
        
        h = layers.ConditionalResidualBlock(
            2 * self.num_features, resample='down', activation=self.activation,
            normalizer=self.normalizer
        )(layer1, labels)
        layer2 = layers.ConditionalResidualBlock(
            2 * self.num_features, resample=None, activation=self.activation, 
            normalizer=self.normalizer
        )(h, labels)
        
        h = layers.ConditionalResidualBlock(
             2 * self.num_features, resample='down', activation=self.activation,
            normalizer=self.normalizer, dilation=2
        )(layer2, labels)
        layer3 = layers.ConditionalResidualBlock(
            2 * self.num_features, resample=None, activation=self.activation,
            normalizer=self.normalizer, dilation=2
        )(h, labels)
        
        h = layers.ConditionalResidualBlock(
            2 * self.num_features, resample='down', activation=self.activation,
            normalizer=self.normalizer, dilation=4
        )(layer3, labels)
        layer4 = layers.ConditionalResidualBlock(
            2 * self.num_features, resample=None, activation=self.activation,
            normalizer=self.normalizer, dilation=4
        )(h, labels)
        
        # U-Net with RefineBlocks
        ref1 = layers.CondRefineBlock(
            layer4.shape[1:2], 2 * self.num_features,
            normalizer=self.normalizer, activation=self.activation,
            interpolation=self.interpolation, start=True
        )([layer4], labels)
        ref2 = layers.CondRefineBlock(
            layer3.shape[1:2], 2 * self.num_features,
            normalizer=self.normalizer, activation=self.activation,
            interpolation=self.interpolation,
        )([layer3, ref1], labels)
        ref3 = layers.CondRefineBlock(
            layer2.shape[1:2], 2 * self.num_features,
            normalizer=self.normalizer, activation=self.activation,
            interpolation=self.interpolation,
        )([layer2, ref2], labels)
        ref4 = layers.CondRefineBlock(
            layer1.shape[1:2], self.num_features,
            normalizer=self.normalizer, activation=self.activation,
            interpolation=self.interpolation, end=True
        )([layer1, ref3], labels)
        h = self.normalizer()(ref4, labels)
        h = self.activation(h)
        h = layers.ncsn_conv(h, x.shape[-1], kernel_size=3)

        used_sigmas = sigmas[labels].reshape(
            (x.shape[0], *([1] * len(x.shape[1:]))))
        
        return h / used_sigmas

# Encoder that returns Gaussian moments
class Encoder(nn.Module):
    features: Sequence[int]
    activation: nn.Module = nn.relu

    @nn.compact
    def __call__(self, x):
        x = x.ravel()
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        y1 = nn.Dense(self.features[-1])(x)
        y2 = nn.Dense(self.features[-1])(x)
        y2 = nn.softplus(y2)

        return y1, y2


# CNN-based encoder that returns Gaussian moments
class CNNEncoder(nn.Module):
    output_dim: int
    activation: nn.Module = nn.relu

    @nn.compact
    def __call__(self, x):
        x = jax.numpy.transpose(x, (1, 0)) # to (batch_size, time, channel)
        x = nn.Conv(features=12, kernel_size=(10,))(x)
        x = self.activation(x)
        x = nn.avg_pool(x, window_shape=(2,), strides=(2,))
        x = nn.Conv(features=12, kernel_size=(10,))(x)
        x = self.activation(x)
        x = nn.avg_pool(x, window_shape=(2,), strides=(2,))
        x = x.ravel()
        x = nn.Dense(features=128)(x)
        x = self.activation(x)
        y1 = nn.Dense(features=self.output_dim)(x).ravel()
        y2 = nn.Dense(features=self.output_dim)(x).ravel()
        y2 = nn.softplus(y2)

        return y1, y2

# Decoder
class Decoder(nn.Module):
    features: Sequence[int]
    activation: nn.Module = nn.relu
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        x = x.ravel()
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1], 
                     use_bias=self.use_bias)(x)

        return x