
import jax
import jax.numpy as jnp
from typing import List
from flax import nnx

from layers.conv_lstm import ConvLSTMCell


class Forecaster(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        forecast_steps: int,
        hidden_dim: int,
        rngs: nnx.Rngs
    ):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.encoder = ConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, rngs=rngs, kernel_size=(3, 3))
        self.decoder = ConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, rngs=rngs, kernel_size=(3, 3))
        self.head = nnx.Conv(hidden_dim, out_features=1, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        B, T_in, H, W, channels = x.shape
        h, c = (
            jnp.zeros((B, H, W, self.encoder.hidden_dim)),
            jnp.zeros((B, H, W, self.encoder.hidden_dim))
        )

        # Encode the input sequence using ConvLSTM
        for t in range(T_in):
            x_t = x[:, t]
            h, c = self.encoder(x_t, hidden_state=(h, c))
        
        outputs = []
        x_t = jnp.zeros((B, H, W, channels))
        for t in range(self.forecast_steps):
            h, c = self.decoder(x_t, hidden_state=(h, c))
            x_t = self.head(h)
            outputs.append(x_t)

        outputs = jnp.stack(outputs, axis=1)  # Stack along the time dimension
        outputs = jax.nn.sigmoid(outputs)
        return outputs
