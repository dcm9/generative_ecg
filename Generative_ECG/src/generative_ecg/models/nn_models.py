from typing import Callable, Sequence

from flax import linen as nn
import jax.nn.initializers as init
import jax.numpy as jnp


import Generative_ECG.src.generative_ecg.models.layer_utils as layers
from Generative_ECG.src.generative_ecg.models.math_utils import get_sigmas

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
        x = jnp.transpose(x, (1, 0)) # to (batch_size, time, channel)
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
        x = jnp.transpose(x, (1, 0)) # to (batch_size, time, channel)
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


class ConditionalInstanceNorm2dPlus(nn.Module):
    """InstanceNorm++ as proposed in the original NCSN paper."""
    num_classes: int = 10
    bias: bool = True
    eps: float = 1e-5
    
    @staticmethod
    def init_embed(key, shape, dtype=jnp.float32, bias=True):
        feature_size = shape[1] // 3
        normal_init = init.normal(0.02)
        normal = normal_init(
            key, (shape[0], 2 * feature_size), dtype=dtype
        ) + 1.
        zero = init.zeros(key, (shape[0], feature_size), dtype=dtype)
        if bias:
            return jnp.concatenate([normal, zero], axis=-1)
        else:
            return normal

    @nn.compact
    def __call__(self, x, y):
        means = jnp.mean(x, axis=1)
        m = jnp.mean(means, axis=-1, keepdims=True)
        v = jnp.var(means, axis=-1, keepdims=True)
        means_plus = (means - m) / jnp.sqrt(v + self.eps)

        h = (x - means[:, None, :]) \
            / jnp.sqrt(jnp.var(x, axis=(1, 2), keepdims=True) + self.eps)
        
        embed = nn.Embed(
            num_embeddings=self.num_classes, 
            features=x.shape[-1] * 3, 
            embedding_init=partial(self.init_embed, bias=self.bias)
        )
        
        if self.bias:
            gamma, alpha, beta = jnp.split(embed(y), 3, axis=-1)
        else:
            gamma, alpha = jnp.split(embed(y), 2, axis=-1)
            beta = jnp.zeros_like(alpha)

        h = h + means_plus[:, None, :] * alpha[:, None, :]
        h = h * gamma[:, None, :]
        h = h + beta[:, None, :]
        
        return h
    

def ncsn_conv(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1., 
              kernel_size=1):
    init_scale = 1e-10 if init_scale == 0 else init_scale
    kernel_init = init.variance_scaling(1/3 * init_scale, 'fan_in', 'uniform')
    kernel_shape = (kernel_size, kernel_size) + (x.shape[-1], out_planes)
    bias_init = lambda key, shape, dtype: \
        kernel_init(key, kernel_shape)[0, 0, 0, :]
    output = nn.Conv(
        out_planes, kernel_size=(kernel_size, kernel_size), 
        strides=(stride, stride), padding='SAME', use_bias=bias, 
        kernel_dilation=(dilation, dilation),
        kernel_init=kernel_init, bias_init=bias_init
    )(x)
    
    return output


class ConvMeanPool(nn.Module):
    output_dim: int
    kernel_size: int = 3
    biases: bool = True

    @nn.compact
    def __call__(self, inputs):
        output = nn.Conv(
            features=self.output_dim, 
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(1, 1), padding='SAME', use_bias=self.biases
        )(inputs)
        output = sum([
            output[:, ::2, :], output[:, 1::2, :],
        ]) / 2.
        
        return output


class ConditionalResidualBlock(nn.Module):
    output_dim: int
    normalizer: Any
    resample: Optional[str] = None
    activation: Any = nn.elu
    dilation: int = 1

    @nn.compact
    def __call__(self, x, y):
        h = self.normalizer()(x, y)
        h = self.activation(h)
        if self.resample == 'down':
            h = ncsn_conv(h, h.shape[-1], dilation=self.dilation, kernel_size=3)
            h = self.normalizer()(h, y)
            h = self.activation(h)
            if self.dilation > 1:
                h = ncsn_conv(h, self.output_dim, dilation=self.dilation,
                              kernel_size=3)
                shortcut = ncsn_conv(x, self.output_dim, dilation=self.dilation,
                                     kernel_size=3)
            else:
                h = ConvMeanPool(output_dim=self.output_dim)(h)
                shortcut = ConvMeanPool(
                    output_dim=self.output_dim, kernel_size=1
                )(x)
        elif self.resample is None:
            h = ncsn_conv(h, self.output_dim, dilation=self.dilation,
                          kernel_size=3)
            h = self.normalizer()(h, y)
            h = self.activation(h)
            h = ncsn_conv(h, self.output_dim, dilation=self.dilation,
                          kernel_size=3)
            if self.output_dim == x.shape[-1]:
                shortcut = x
            else:
                if self.dilation > 1:
                    shortcut = ncsn_conv(x, self.output_dim, 
                                         dilation=self.dilation,
                                         kernel_size=3)
                else:
                    shortcut = ncsn_conv(x, self.output_dim, kernel_size=1)

        return h + shortcut
    

class CondRCUBlock(nn.Module):
    features: int
    n_blocks: int
    n_stages: int
    normalizer: Any
    activation: Any = nn.elu

    @nn.compact
    def __call__(self, x, y):
        for _ in range(self.n_blocks):
            residual = x
            for _ in range(self.n_stages):
                x = self.normalizer()(x, y)
                x = self.activation(x)
                x = ncsn_conv(x, self.features, stride=1, bias=False,
                              kernel_size=3)
            x += residual
        
        return x
    

class CondMSFBlock(nn.Module):
    shape: Sequence[int]
    features: int
    normalizer: Any
    interpolation: str = 'bilinear'

    @nn.compact
    def __call__(self, xs, y):
        sums = jnp.zeros((xs[0].shape[0], *self.shape, self.features))
        for i in range(len(xs)):
            h = self.normalizer()(xs[i], y)
            h = ncsn_conv(h, self.features, stride=1, bias=True, kernel_size=3)
            if self.interpolation == 'bilinear':
                h = jax.image.resize(
                    h, (h.shape[0], *self.shape, h.shape[-1]), 'bilinear'
                )
            elif self.interpolation == 'nearest_neighbor':
                h = jax.image.resize(
                    h, (h.shape[0], *self.shape, h.shape[-1]), 'nearest'
                )
            else:
                raise ValueError(
                    f'Interpolation {self.interpolation} does not exist'
                )
            sums = sums + h
            
        return sums
    

class CondCRPBlock(nn.Module):
    features: int
    n_stages: int
    normalizer: Any
    activation: Any = nn.elu

    @nn.compact
    def __call__(self, x, y):
        x = self.activation(x)
        path = x
        for _ in range(self.n_stages):
            path = self.normalizer()(path, y)
            path = nn.avg_pool(
                path, window_shape=(5, 5), strides=(1, 1), padding='SAME'
            )
            path = ncsn_conv(path, self.features, stride=1, bias=False,
                             kernel_size=3)
            x = path + x
            
        return x

    
class CondRefineBlock(nn.Module):
    output_shape: Sequence[int]
    features: int
    normalizer: Any
    activation: Any = nn.elu
    interpolation: str = 'bilinear'
    start: bool = False
    end: bool = False

    @nn.compact
    def __call__(self, xs, y):
        hs = []
        for i in range(len(xs)):
            h = CondRCUBlock(
                n_blocks=2, n_stages=2, activation=self.activation, 
                normalizer=self.normalizer, features=xs[i].shape[-1]
            )(xs[i], y)
            hs.append(h)

        if not self.start:
            h = CondMSFBlock(
                features=self.features, interpolation=self.interpolation,
                normalizer=self.normalizer, shape=self.output_shape
            )(hs, y)
        else:
            h = hs[0]

        h = CondCRPBlock(
            features=self.features, n_stages=2, activation=self.activation,
            normalizer=self.normalizer
        )(h, y)
        rcu_block_output = partial(
            CondRCUBlock, features=self.features, n_blocks=3 if self.end else 1,
            n_stages=2, activation=self.activation, normalizer=self.normalizer
        )
        h = rcu_block_output()(h, y)
        
        return h
