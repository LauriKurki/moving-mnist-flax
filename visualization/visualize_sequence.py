import os
import jax
import matplotlib.pyplot as plt


def visualize_sequence(x, y, y_hat, title="Sequence", save_path="figures/sequence_visualization.png"):
    fig, axs = plt.subplots(3, 10, figsize=(15, 4))
    fig.subplots_adjust(hspace=0.5, wspace=0.1)
    for i in range(10):
        axs[0, i].imshow(x[..., i], cmap="gray")
        axs[1, i].imshow(y[..., i], cmap="gray")
        axs[2, i].imshow(y_hat[..., i], cmap="gray")
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
        axs[2, i].set_xticks([])
        axs[2, i].set_yticks([])

    # Set titles for each row
    axs[0, 0].set_ylabel("Input Sequence", fontsize=12)
    axs[1, 0].set_ylabel("Output Sequence", fontsize=12)
    axs[2, 0].set_ylabel("Predicted Sequence", fontsize=12)
    axs[0, 0].set_title("Input", fontsize=14)
    axs[1, 0].set_title("Output", fontsize=14)
    axs[2, 0].set_title("Predicted", fontsize=14)

    # Add sequence number labels
    for i in range(10):
        axs[0, i].set_xlabel(f"i={i+1}", fontsize=10)
        axs[1, i].set_xlabel(f"i={i+1}", fontsize=10)
        axs[2, i].set_xlabel(f"i={i+1}", fontsize=10)
    plt.suptitle(title)
    plt.savefig(save_path)
    plt.close(fig)


def make_predictions(model, test_dl, num_predictions: int = 5, save_path: str = "figures/predictions/"):
    counter = 0
    os.makedirs(save_path, exist_ok=True)
    for x, y in test_dl:
        yhat = model(x)
        yhat = jax.nn.sigmoid(yhat) 

        for i in range(x.shape[0]):
            xi = x[i]
            yi = y[i]
            yhat_i = yhat[i]

            # Visualize the sequence
            visualize_sequence(
                xi, yi, yhat_i,
                save_path=os.path.join(save_path, f"sample_{counter:02}.png")
            )
            counter += 1
        if counter >= num_predictions:
            break


def plot_f1_scores(train_f1, test_f1, save_path="figures/f1_scores.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_f1, label='Train F1 Score', marker='o')
    plt.plot(test_f1, label='Test F1 Score', marker='x')
    plt.title('F1 Scores as a Function of Leadtime')
    plt.xlabel('Leadtime')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()
