import jax.numpy as jnp
from flax import nnx

from typing import List, Optional


class ConvLSTMCell(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
        kernel_size: tuple = (3, 3),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.conv = nnx.Conv(
            in_features=input_dim + hidden_dim,
            out_features=hidden_dim * 4,  # i, f, o, g
            kernel_size=self.kernel_size,
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, x, hidden_state: Optional[tuple] = None):
        if hidden_state is None:
            h_current, c_current = self.init_hidden_state(x.shape[0], x.shape[2], x.shape[3])
        else:
            h_current, c_current = hidden_state

        combined = jnp.concatenate([x, h_current], axis=-1) # concatenate along the channel dimension
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = jnp.split(combined_conv, 4, axis=-1)
        i = nnx.sigmoid(cc_i)
        f = nnx.sigmoid(cc_f)
        o = nnx.sigmoid(cc_o)
        g = jnp.tanh(cc_g)

        c_next = f * c_current + i * g
        h_next = o * jnp.tanh(c_next)

        return h_next, c_next
    
    def init_hidden_state(self, batch_size, height, width):
        h = jnp.zeros((batch_size, height, width, self.hidden_dim))
        c = jnp.zeros((batch_size, height, width, self.hidden_dim))
        return h, c


class ConvLSTM(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: List[int],
        rngs: nnx.Rngs,
        kernel_size: tuple = (3, 3),
        num_layers: int = 1,
        return_last_only: bool = False,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.return_last_only = return_last_only

        self.cells = [
            ConvLSTMCell(
                input_dim=self.input_dim if i == 0 else self.hidden_dim[i-1],
                hidden_dim=self.hidden_dim[i],
                kernel_size=self.kernel_size,
                rngs=rngs
            ) for i in range(self.num_layers)
        ]

    def __call__(self, x, hidden_state=None):
        b, seq_len, h, w, ch = x.shape
        if hidden_state is None:
            hidden_state = self.init_hidden_state(b, h, w)
        cur_input = x

        layer_output_list = []
        layer_state_list = []

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            outputs = []
            for t in range(seq_len):
                print(cur_input[:, t].shape, h.shape, c.shape)
                print(self.cells[layer_idx])
                h, c = self.cells[layer_idx](cur_input[:, t], (h, c))
                outputs.append(h)

            layer_output = jnp.stack(outputs, axis=1)
            cur_input = layer_output

            layer_output_list.append(layer_output)
            layer_state_list.append((h, c))

        if self.return_last_only:
            return layer_output_list[-1], layer_state_list[-1]

        return layer_output_list, layer_state_list

    def init_hidden_state(self, batch_size, height, width):
        init_states = []
        for cell in self.cells:
            init_states.append(cell.init_hidden_state(batch_size, height, width))
        return init_states
