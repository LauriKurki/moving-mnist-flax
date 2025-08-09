import os
import tqdm
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from models.forecaster import Forecaster
from metrics.metrics import F1PerLeadtime
from data_loading import get_dataloaders
from visualization import visualize_sequence


from typing import Tuple


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train a Forecaster model on Moving MNIST dataset.")
    parser.add_argument("--epochs", "-n", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size for training and evaluation")
    parser.add_argument("--train-steps", "-t", type=int, default=1e9, help="Number of training steps")
    parser.add_argument("--eval-steps", "-e", type=int, default=1e9, help="Number of evaluation steps")

    return parser.parse_args()

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
    metrics.update(loss=loss, y_true=batch[1], y_pred=y_hat)
    return loss

@nnx.jit
def eval_step(
    model: nnx.Module,
    metrics: nnx.MultiMetric,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    loss, y_hat = loss_fn(model, batch)  # Compute loss and predictions
    metrics.update(loss=loss, y_true=batch[1], y_pred=y_hat)
    return loss


def main():
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    train_steps = args.train_steps
    eval_steps = args.eval_steps

    metrics_history = {
        'train_loss': [],
        'train_f1': [],
        'test_loss': [],
        'test_f1': [],
    }
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        f1=F1PerLeadtime(10)
    )
    # Define the dataset and dataloaders
    train_dl, test_dl = get_dataloaders(batch_size=batch_size)

    # Define the model
    model = Forecaster(
        input_timesteps=10,
        forecast_timesteps=10,
        base_channels=32,
        depth=4,
        rngs=nnx.Rngs(0),
    )
    learning_rate = 0.0003
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=learning_rate))

    checkpointer = ocp.StandardCheckpointer()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = Path(current_dir) / "weights"
    ocp.test_utils.erase_and_create_empty(ckpt_dir)
    #ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):

        # Training
        model.train()
        with tqdm.tqdm(train_dl, total=len(train_dl), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for i, batch in enumerate(pbar):
                loss = train_step(model, optimizer, metrics, batch)
                pbar.set_postfix({'loss': float(loss)})
                if i >= train_steps:
                    break

        for metric, value in metrics.compute().items():  # compute metrics
            metrics_history[f'train_{metric}'].append(value)  # record metrics
            metrics.reset()  # reset metrics for test set

        # Compute metrics on the test set after each training epoch
        model.eval()
        with tqdm.tqdm(test_dl, total=len(test_dl), desc="Evaluating") as pbar:
            for i, batch in enumerate(pbar):
                loss = eval_step(model, metrics, batch)
                pbar.set_postfix({'loss': float(loss)})
                if i >= eval_steps:
                    break

        # Log test metrics
        for metric, value in metrics.compute().items():
            metrics_history[f'test_{metric}'].append(value)
            metrics.reset()  # reset metrics for next training epoch

        print(f"train epoch: {epoch}, f1: {metrics_history['train_f1'][-1]}")
        print(f"test epoch: {epoch}, f1: {metrics_history['test_f1'][-1]}")

    # Save the model state
    _, state = nnx.split(model)
    dict_state = nnx.to_pure_dict(state)
    checkpointer.save(ckpt_dir / f'state_trained', dict_state)

    # At the end of training, plot f1 score and save predictions
    visualize_sequence.plot_f1_scores(metrics_history['train_f1'][-1], metrics_history['test_f1'][-1], save_path="figures/f1_scores.png")
    visualize_sequence.make_predictions(
        model, test_dl, num_predictions=30, save_path="figures/predictions/"
    )


if __name__ == "__main__":
    main()
