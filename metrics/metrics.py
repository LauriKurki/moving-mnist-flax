import jax
import jax.numpy as jnp
import flax.nnx as nnx

class F1PerLeadtime(nnx.Metric):
    def __init__(self, num_timesteps: int, eps: float = 1e-7):
        self.num_timesteps = num_timesteps
        self.eps = eps
        
        self.tp = jnp.zeros((num_timesteps,), dtype=jnp.float32)
        self.fp = jnp.zeros((num_timesteps,), dtype=jnp.float32)
        self.fn = jnp.zeros((num_timesteps,), dtype=jnp.float32)
    
    def update(self, *, y_true, y_pred, **_):
        """
        y_true, y_pred: (B, H, W, T) with binary labels {0,1}
        """
        y_pred = jax.nn.sigmoid(y_pred) > 0.5  # Convert logits to binary predictions
        for t in range(self.num_timesteps):
            yt = y_true[..., t].reshape(-1)
            yp = y_pred[..., t].reshape(-1)

            tp_t = jnp.sum((yp == 1) & (yt == 1))
            fp_t = jnp.sum((yp == 1) & (yt == 0))
            fn_t = jnp.sum((yp == 0) & (yt == 1))

            self.tp = self.tp.at[t].add(tp_t)
            self.fp = self.fp.at[t].add(fp_t)
            self.fn = self.fn.at[t].add(fn_t)
    
    def compute(self):
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        return f1
    
    def reset(self):
        self.tp = jnp.zeros((self.num_timesteps,), dtype=jnp.float32)
        self.fp = jnp.zeros((self.num_timesteps,), dtype=jnp.float32)
        self.fn = jnp.zeros((self.num_timesteps,), dtype=jnp.float32)
