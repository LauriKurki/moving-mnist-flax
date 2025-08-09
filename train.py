import os
import tqdm

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from models.forecaster import Forecaster
from data_loading import get_dataloaders
from visualization import visualize_sequence

from torch.utils.data import DataLoader
from typing import Tuple

def loss_fn(
    model: nnx.Module,
    batch: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x, y = batch
    y_hat = model(x)
    # Loss is binary cross-entropy for each pixel in the output sequence
    loss = optax.losses.sigmoid_binary_cross_entropy(y_hat, y).mean()
    return loss, y_hat

@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    (loss, y_hat), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
    optimizer.update(grads)
    metrics.update(loss=loss, logits=y_hat[..., None], labels=batch[1])
    return loss

@nnx.jit
def eval_step(
    model: nnx.Module,
    metrics: nnx.MultiMetric,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    loss, y_hat = loss_fn(model, batch)  # Compute loss and predictions
    metrics.update(loss=loss, logits=y_hat[..., None], labels=batch[1])
    return loss


def main():
    metrics_history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
    }
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        accuracy=nnx.metrics.Accuracy(),
    )
    # Define the dataset and dataloaders
    train_dl, test_dl = get_dataloaders(batch_size=8)

    # Define the model
    model = Forecaster(
        input_dim=1,
        forecast_steps=10,
        hidden_dim=32,
        rngs=nnx.Rngs(0),
    )
    learning_rate = 0.0003
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=learning_rate))

    num_epochs = 10
    for epoch in range(num_epochs):
        # Training
        with tqdm.tqdm(train_dl, total=len(train_dl), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in pbar:
                loss = train_step(model, optimizer, metrics, batch)
                pbar.set_postfix({'loss': float(loss)})

        for metric, value in metrics.compute().items():  # compute metrics
            metrics_history[f'train_{metric}'].append(value)  # record metrics
            metrics.reset()  # reset metrics for test set

        # Compute metrics on the test set after each training epoch
        with tqdm.tqdm(test_dl, total=len(test_dl), desc="Evaluating") as pbar:
            for batch in pbar:
                loss = eval_step(model, metrics, batch)
                pbar.set_postfix({'loss': float(loss)})

        # Log test metrics
        for metric, value in metrics.compute().items():
            metrics_history[f'test_{metric}'].append(value)
            metrics.reset()  # reset metrics for next training epoch

        print(
            f"train epoch: {epoch}, "
            f"loss: {metrics_history['train_loss'][-1]}, "
            f"accuracy: {metrics_history['train_accuracy'][-1] * 100}"
        )
        print(
            f"test epoch: {epoch}, "
            f"loss: {metrics_history['test_loss'][-1]}, "
            f"accuracy: {metrics_history['test_accuracy'][-1] * 100}"
        )

        # At the end of each epoch, you can save the model or visualize results
        x, y = next(iter(test_dl)) # (bs, height, width, seq_len), (bs, height, width, 1)
        yhat = model(x) # (bs, height, width, 1)
        epoch_save_path = f"figures/epoch_{epoch}/"
        os.makedirs(epoch_save_path, exist_ok=True)
        for i in range(x.shape[0]):
            xi = x[i]
            yi = y[i]
            yhat_i = yhat[i]

            # Visualize the sequence
            visualize_sequence.visualize_sequence(
                xi, yi, yhat_i,
                title=f"Epoch {epoch} - Sample {i}",
                save_path=os.path.join(epoch_save_path, f"sample_{i}.png")
            )


if __name__ == "__main__":
    main()
