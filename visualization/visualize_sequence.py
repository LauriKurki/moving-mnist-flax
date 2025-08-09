import matplotlib.pyplot as plt

import data_loading

def visualize_sequence(x, y, y_hat, title="Sequence", save_path="figures/sequence_visualization.png"):
    fig, axs = plt.subplots(3, 10, figsize=(15, 4))
    fig.subplots_adjust(hspace=0.5, wspace=0.1)
    print(x.shape, y.shape, y_hat.shape)
    for i in range(10):
        axs[0, i].imshow(x[i], cmap="gray")
        axs[1, i].imshow(y[i], cmap="gray")
        axs[2, i].imshow(y_hat[i], cmap="gray")
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

def load_sample():
    dl, _ = data_loading.get_dataloaders(batch_size=None)
    x, y = next(iter(dl))
    return x, y

if __name__ == "__main__":
    x, y = load_sample()
    visualize_sequence(x, y, title="Sample Sequence Visualization")
