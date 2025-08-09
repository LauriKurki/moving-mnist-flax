
import jax
import jax.numpy as jnp
from typing import List
from flax import nnx

from layers.layers import ResidualBlock, DownBlock, UpBlock

class Forecaster(nnx.Module):
    def __init__(self, input_timesteps, forecast_timesteps, base_channels=32, depth=4, rngs=None):
        # Encoder
        self.encoders = []
        ch = base_channels
        self.encoders.append(ResidualBlock(input_timesteps, ch, rngs=rngs))

        self.down_blocks = []
        for _ in range(depth - 1):
            self.down_blocks.append(DownBlock(ch, ch * 2, rngs=rngs))
            ch *= 2
        
        # Bottleneck
        self.bottleneck = ResidualBlock(ch, ch * 2, rngs=rngs)
        ch *= 2
        
        # Decoder
        self.up_blocks = []
        for _ in range(depth - 1):
            self.up_blocks.append(UpBlock(ch, ch // 2, rngs=rngs))
            ch //= 2
        
        self.final_up = UpBlock(ch, base_channels, rngs=rngs)
        
        # Output head
        self.out_conv = nnx.Conv(
            in_features=base_channels,
            out_features=forecast_timesteps,
            kernel_size=(1, 1),
            padding="SAME",
            use_bias=True,
            rngs=rngs
        )

    def __call__(self, x):
        skips = []
        # First encoder block
        x = self.encoders[0](x)
        skips.append(x)
        
        # Down path
        for down in self.down_blocks:
            x, skip = down(x)
            skips.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Up path
        for up in self.up_blocks:
            skip = skips.pop()
            x = up(x, skip)
        
        # Final up
        skip = skips.pop()
        x = self.final_up(x, skip)
        
        # Output
        return self.out_conv(x)
