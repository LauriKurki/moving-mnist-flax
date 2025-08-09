import unittest

import jax.numpy as jnp
from flax import nnx
from models.forecaster import Forecaster


class TestForecaster(unittest.TestCase):
    def setUp(self):
        rngs = nnx.Rngs(0)
        self.forecast_steps = 10
        self.model = Forecaster(
            input_timesteps=10,
            forecast_steps=self.forecast_steps,
            base_channels=32,
            depth=4,
            rngs=rngs
        )

    def test_forward(self):
        x = jnp.ones((8, 32, 32, 10))  # (batch, height, width, seq_len)
        y = self.model(x)
        self.assertEqual(y.shape, (8, 32, 32, self.forecast_steps),
                         "Output shape should match forecast steps")


if __name__ == "__main__":
    unittest.main()
