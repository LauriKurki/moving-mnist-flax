import unittest

import jax.numpy as jnp
from flax import nnx
from models.forecaster import Forecaster


class TestForecaster(unittest.TestCase):
    def setUp(self):
        rngs = nnx.Rngs(0)
        self.forecast_steps = 10
        self.model = Forecaster(
            input_dim=1,
            forecast_steps=self.forecast_steps,
            hidden_dim=32,
            rngs=rngs
        )

    def test_forward(self):
        x = jnp.ones((8, 10, 32, 32, 1))  # (batch, seq_len, height, width, channels)
        y = self.model(x)
        self.assertEqual(y.shape, (8, 10, 32, 32, 1))


if __name__ == "__main__":
    unittest.main()
