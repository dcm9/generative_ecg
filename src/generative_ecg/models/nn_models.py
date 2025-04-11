from typing import Callable, Sequence

from flax import linen 
import jax.nn.initializers 
import jax.numpy


from . import layer_utils 
from .math_utils import get_sigmas

from functools import partial
from typing import Any, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

class ECGConv(linen.Module):
    """
    Convolutional Neural Network for ECG data processing.
    
    Attributes:
        tmax: Maximum time steps in the ECG signal
        n_channels: Number of ECG channels (leads)
        n_layers_conv: Number of convolutional layers
        n_layers_dense: Number of dense layers after convolution
        n_outputs: Number of output dimensions
        hidden_dim: Hidden dimension size (default: 64)
        kernel_size: Size of convolutional kernels (default: 3)
    """
    tmax: int
    n_channels: int
    n_layers_conv: int
    n_layers_dense: int
    n_outputs: int
    hidden_dim: int = 64
    kernel_size: int = 3
    
    @nn.compact
    def __call__(self, x, training=True):
        # Input shape: (batch_size, n_channels, tmax)
        # Reshape to (batch_size, tmax, n_channels) for conv layers
        x = jnp.transpose(x, (0, 2, 1))
        
        # Initial feature dimension
        features = self.hidden_dim
        
        # Convolutional layers
        for i in range(self.n_layers_conv):
            x = nn.Conv(
                features=features,
                kernel_size=(self.kernel_size,),
                strides=(1,),
                padding="SAME",
                name=f"conv_{i}"
            )(x)
            x = nn.BatchNorm(use_running_average=not training, name=f"bn_conv_{i}")(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2,), strides=(2,), padding="VALID")
            
            # Double features for each layer
            features *= 2
        
        # Global average pooling
        x = jnp.mean(x, axis=1)
        
        # Dense layers
        for i in range(self.n_layers_dense):
            x = nn.Dense(features=features, name=f"dense_{i}")(x)
            x = nn.BatchNorm(use_running_average=not training, name=f"bn_dense_{i}")(x)
            x = nn.relu(x)
            
            # Halve features for each layer
            features //= 2
            # Ensure we don't go below the output dimension
            features = max(features, self.n_outputs)
        
        # Output layer
        x = nn.Dense(features=self.n_outputs, name="output")(x)
        
        return x

class MLP(linen.Module):
    features: Sequence[int]
    activation: linen.Module = linen.softplus

    @linen.compact
    def __call__(self, x):
        x = x.ravel()
        for feat in self.features[:-1]:
            x = self.activation(linen.Dense(feat)(x))
        x = linen.Dense(self.features[-1])(x)

        return x
    
    
class CNN(linen.Module):
    output_dim: int
    activation: linen.Module = linen.relu
    
    @linen.compact
    def __call__(self, x):
        x = jax.numpy.transpose(x, (1, 0)) # to (batch_size, time, channel)
        x = linen.Conv(features=12, kernel_size=(10,))(x)
        x = self.activation(x)
        x = linen.avg_pool(x, window_shape=(2,), strides=(2,))
        x = linen.Conv(features=12, kernel_size=(10,))(x)
        x = self.activation(x)
        x = linen.avg_pool(x, window_shape=(2,), strides=(2,))
        x = x.ravel()
        x = linen.Dense(features=128)(x)
        x = self.activation(x)
        x = linen.Dense(features=self.output_dim)(x).ravel()
        
        return x


class NCSN(linen.Module):
    activation: linen.Module = linen.elu
    num_features: int = 64
    normalizer: linen.Module = layer_utils.ConditionalInstanceNorm2dPlus
    get_sigmas: Callable = get_sigmas
    interpolation: str = 'bilinear'

    @linen.compact
    def __call__(self, x, labels, train=True):
        sigmas = self.get_sigmas()
        
        # h = 2 * x - 1. # [0, 1] -> [-1, 1]
        h = x
        h = layer_utils.ncsn_conv(h, self.num_features, stride=1, bias=True,
                             kernel_size=3)
        # print(f"labels: {labels}")
        
        # ResNet backbone
        h = layer_utils.ConditionalResidualBlock(
            self.num_features, resample=None, activation=self.activation, 
            normalizer=self.normalizer
        )(h, labels)
        layer1 = layer_utils.ConditionalResidualBlock(
            self.num_features, resample=None, activation=self.activation, 
            normalizer=self.normalizer
        )(h, labels)
        
        h = layer_utils.ConditionalResidualBlock(
            2 * self.num_features, resample='down', activation=self.activation,
            normalizer=self.normalizer
        )(layer1, labels)
        layer2 = layer_utils.ConditionalResidualBlock(
            2 * self.num_features, resample=None, activation=self.activation, 
            normalizer=self.normalizer
        )(h, labels)
        
        h = layer_utils.ConditionalResidualBlock(
             2 * self.num_features, resample='down', activation=self.activation,
            normalizer=self.normalizer, dilation=2
        )(layer2, labels)
        layer3 = layer_utils.ConditionalResidualBlock(
            2 * self.num_features, resample=None, activation=self.activation,
            normalizer=self.normalizer, dilation=2
        )(h, labels)
        
        h = layer_utils.ConditionalResidualBlock(
            2 * self.num_features, resample='down', activation=self.activation,
            normalizer=self.normalizer, dilation=4
        )(layer3, labels)
        layer4 = layer_utils.ConditionalResidualBlock(
            2 * self.num_features, resample=None, activation=self.activation,
            normalizer=self.normalizer, dilation=4
        )(h, labels)
        
        # U-Net with RefineBlocks
        ref1 = layer_utils.CondRefineBlock(
            layer4.shape[1:2], 2 * self.num_features,
            normalizer=self.normalizer, activation=self.activation,
            interpolation=self.interpolation, start=True
        )([layer4], labels)
        ref2 = layer_utils.CondRefineBlock(
            layer3.shape[1:2], 2 * self.num_features,
            normalizer=self.normalizer, activation=self.activation,
            interpolation=self.interpolation,
        )([layer3, ref1], labels)
        ref3 = layer_utils.CondRefineBlock(
            layer2.shape[1:2], 2 * self.num_features,
            normalizer=self.normalizer, activation=self.activation,
            interpolation=self.interpolation,
        )([layer2, ref2], labels)
        ref4 = layer_utils.CondRefineBlock(
            layer1.shape[1:2], self.num_features,
            normalizer=self.normalizer, activation=self.activation,
            interpolation=self.interpolation, end=True
        )([layer1, ref3], labels)
        h = self.normalizer()(ref4, labels)
        h = self.activation(h)
        h = layer_utils.ncsn_conv(h, x.shape[-1], kernel_size=3)

        used_sigmas = sigmas[labels].reshape(
            (x.shape[0], *([1] * len(x.shape[1:]))))
        
        return h / used_sigmas

# Encoder that returns Gaussian moments
class Encoder(linen.Module):
    features: Sequence[int]
    activation: linen.Module = linen.relu

    @linen.compact
    def __call__(self, x):
        x = x.ravel()
        for feat in self.features[:-1]:
            x = self.activation(linen.Dense(feat)(x))
        y1 = linen.Dense(self.features[-1])(x)
        y2 = linen.Dense(self.features[-1])(x)
        y2 = linen.softplus(y2)

        return y1, y2


# CNN-based encoder that returns Gaussian moments
class CNNEncoder(linen.Module):
    output_dim: int
    activation: linen.Module = linen.relu

    @linen.compact
    def __call__(self, x):
        x = jax.numpy.transpose(x, (1, 0)) # to (batch_size, time, channel)
        x = linen.Conv(features=12, kernel_size=(10,))(x)
        x = self.activation(x)
        x = linen.avg_pool(x, window_shape=(2,), strides=(2,))
        x = linen.Conv(features=12, kernel_size=(10,))(x)
        x = self.activation(x)
        x = linen.avg_pool(x, window_shape=(2,), strides=(2,))
        x = x.ravel()
        x = linen.Dense(features=128)(x)
        x = self.activation(x)
        y1 = linen.Dense(features=self.output_dim)(x).ravel()
        y2 = linen.Dense(features=self.output_dim)(x).ravel()
        y2 = linen.softplus(y2)

        return y1, y2

# Decoder
class Decoder(linen.Module):
    features: Sequence[int]
    activation: linen.Module = linen.relu
    use_bias: bool = True

    @linen.compact
    def __call__(self, x):
        x = x.ravel()
        for feat in self.features[:-1]:
            x = self.activation(linen.Dense(feat)(x))
        x = linen.Dense(self.features[-1], 
                     use_bias=self.use_bias)(x)

        return x