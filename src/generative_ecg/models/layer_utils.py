from functools import partial
from typing import Any, Optional, Sequence

import flax.linen 
import jax
import jax.nn.initializers 
import jax.numpy

class ConditionalInstanceNorm2dPlus(flax.linen.Module):
    """InstanceNorm++ as proposed in the original NCSN paper."""
    num_classes: int = 10
    bias: bool = True
    eps: float = 1e-5
    
    @staticmethod
    def init_embed(key, shape, dtype=jax.numpy.float32, bias=True):
        feature_size = shape[1] // 3
        normal_init = jax.nn.initializers.normal(0.02)
        normal = normal_init(
            key, (shape[0], 2 * feature_size), dtype=dtype
        ) + 1.
        zero = jax.nn.initializers.zeros(key, (shape[0], feature_size), dtype=dtype)
        if bias:
            return jax.numpy.concatenate([normal, zero], axis=-1)
        else:
            return normal

    @flax.linen.compact
    def __call__(self, x, y):
        means = jax.numpy.mean(x, axis=1)
        m = jax.numpy.mean(means, axis=-1, keepdims=True)
        v = jax.numpy.var(means, axis=-1, keepdims=True)
        means_plus = (means - m) / jax.numpy.sqrt(v + self.eps)

        h = (x - means[:, None, :]) \
            / jax.numpy.sqrt(jax.numpy.var(x, axis=(1, 2), keepdims=True) + self.eps)
        
        embed = flax.linen.Embed(
            num_embeddings=self.num_classes, 
            features=x.shape[-1] * 3, 
            embedding_init=partial(self.init_embed, bias=self.bias)
        )
        
        if self.bias:
            gamma, alpha, beta = jax.numpy.split(embed(y), 3, axis=-1)
        else:
            gamma, alpha = jax.numpy.split(embed(y), 2, axis=-1)
            beta = jax.numpy.zeros_like(alpha)

        h = h + means_plus[:, None, :] * alpha[:, None, :]
        h = h * gamma[:, None, :]
        h = h + beta[:, None, :]
        
        return h
    

def ncsn_conv(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1., 
              kernel_size=1):
    init_scale = 1e-10 if init_scale == 0 else init_scale
    kernel_init = jax.nn.initializers.variance_scaling(1/3 * init_scale, 'fan_in', 'uniform')
    kernel_shape = (kernel_size, kernel_size) + (x.shape[-1], out_planes)
    bias_init = lambda key, shape, dtype: \
        kernel_init(key, kernel_shape)[0, 0, 0, :]
    output = flax.linen.Conv(
        out_planes, kernel_size=(kernel_size, kernel_size), 
        strides=(stride, stride), padding='SAME', use_bias=bias, 
        kernel_dilation=(dilation, dilation),
        kernel_init=kernel_init, bias_init=bias_init
    )(x)
    
    return output


class ConvMeanPool(flax.linen.Module):
    output_dim: int
    kernel_size: int = 3
    biases: bool = True

    @flax.linen.compact
    def __call__(self, inputs):
        output = flax.linen.Conv(
            features=self.output_dim, 
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(1, 1), padding='SAME', use_bias=self.biases
        )(inputs)
        output = sum([
            output[:, ::2, :], output[:, 1::2, :],
        ]) / 2.
        
        return output


class ConditionalResidualBlock(flax.linen.Module):
    output_dim: int
    normalizer: Any
    resample: Optional[str] = None
    activation: Any = flax.linen.elu
    dilation: int = 1

    @flax.linen.compact
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
    

class CondRCUBlock(flax.linen.Module):
    features: int
    n_blocks: int
    n_stages: int
    normalizer: Any
    activation: Any = flax.linen.elu

    @flax.linen.compact
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
    

class CondMSFBlock(flax.linen.Module):
    shape: Sequence[int]
    features: int
    normalizer: Any
    interpolation: str = 'bilinear'

    @flax.linen.compact
    def __call__(self, xs, y):
        sums = jax.numpy.zeros((xs[0].shape[0], *self.shape, self.features))
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
    

class CondCRPBlock(flax.linen.Module):
    features: int
    n_stages: int
    normalizer: Any
    activation: Any = flax.linen.elu

    @flax.linen.compact
    def __call__(self, x, y):
        x = self.activation(x)
        path = x
        for _ in range(self.n_stages):
            path = self.normalizer()(path, y)
            path = flax.linen.avg_pool(
                path, window_shape=(5, 5), strides=(1, 1), padding='SAME'
            )
            path = ncsn_conv(path, self.features, stride=1, bias=False,
                             kernel_size=3)
            x = path + x
            
        return x

    
class CondRefineBlock(flax.linen.Module):
    output_shape: Sequence[int]
    features: int
    normalizer: Any
    activation: Any = flax.linen.elu
    interpolation: str = 'bilinear'
    start: bool = False
    end: bool = False

    @flax.linen.compact
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
