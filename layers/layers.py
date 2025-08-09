import jax
import jax.numpy as jnp
from flax import nnx

class ConvLayer(nnx.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, rngs=None):
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride),
            padding="SAME",
            use_bias=False,
            rngs=rngs
        )
        self.norm = nnx.BatchNorm(num_features=out_channels, rngs=rngs)
        self.act = nnx.relu

    def __call__(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class ResidualBlock(nnx.Module):
    def __init__(self, in_channels, out_channels, rngs=None):
        self.conv1 = ConvLayer(in_channels, out_channels, rngs=rngs)
        self.conv2 = ConvLayer(out_channels, out_channels, rngs=rngs)

        if in_channels != out_channels:
            self.skip_conv = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(1, 1),
                padding="SAME",
                use_bias=False,
                rngs=rngs
            )
        else:
            self.skip_conv = None

    def __call__(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
        
        return nnx.relu(out + identity)


class DownBlock(nnx.Module):
    def __init__(self, in_channels, out_channels, rngs=None):
        self.res_block = ResidualBlock(in_channels, out_channels, rngs=rngs)
        self.downsample = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            rngs=rngs
        )

    def __call__(self, x):
        x = self.res_block(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nnx.Module):
    def __init__(self, in_channels, out_channels, rngs=None):
        self.conv_up = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            rngs=rngs
        )
        self.res_block = ResidualBlock(out_channels * 2, out_channels, rngs=rngs)  # *2 due to concat

    def __call__(self, x, skip):
        # Bilinear upsample
        x = jax.image.resize(x, (x.shape[0], skip.shape[1], skip.shape[2], x.shape[3]), method="bilinear")
        x = self.conv_up(x)
        x = jnp.concatenate([x, skip], axis=-1)  # channels-last concat
        x = self.res_block(x)
        return x
