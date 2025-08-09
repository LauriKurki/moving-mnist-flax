import unittest
import jax.numpy as jnp
from flax import nnx
from layers.conv_lstm import ConvLSTM


class TestConvLSTM(unittest.TestCase):

    def setUp(self):
        rngs = nnx.Rngs(0)
        self.encoder = ConvLSTM(
            input_dim=1,
            hidden_dim=[4, 8, 16],
            kernel_size=(3, 3),
            num_layers=3,
            return_last_only=True,
            rngs=rngs
        )
        self.decoder = ConvLSTM(
            input_dim=16,
            hidden_dim=[16, 8, 4],
            kernel_size=(3, 3),
            num_layers=3,
            return_last_only=True,
            rngs=rngs
        )

    def test_forward(self):
        x = jnp.ones((2, 10, 64, 64, 1))  # (batch, seq_len, height, width, channels)
        output, _ = self.encoder(x)
        print("Output shape:", output.shape)
        self.assertEqual(output.shape, (2, 10, 64, 64, 64))

    def test_encoder_decoder(self):
        x = jnp.ones((2, 10, 64, 64, 1))
        x, (h, c) = self.encoder(x)
        print("ASDASDASD", x.shape)

        x_t = jnp.zeros((2, 10, 64, 64, 16))
        x, _ = self.decoder(x_t, hidden_state=(h, c))
        print("Decoder output shape:", x.shape)


        

if __name__ == "__main__":
    unittest.main()
