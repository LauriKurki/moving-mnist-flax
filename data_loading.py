import numpy as np
from torch.utils.data import DataLoader, Dataset

def jnp_collate(batch):
    """
    Custom collate function to handle JAX arrays in DataLoader.
    """
    return tuple(np.stack(x, axis=0) for x in zip(*batch))

class MovingMNISTDataset(Dataset):
    def __init__(
        self,
        fname: str,
        indices: tuple,
        train_seq_length: int = 10
    ):
        self.fname = fname
        self.data = np.load(self.fname)[:, indices[0]:indices[1]]
        self.length = indices[1] - indices[0]
        self.train_seq_length = train_seq_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[:self.train_seq_length, idx, ..., None]  # (H, W, seq)
        y = self.data[self.train_seq_length:, idx, ..., None] # (H, W, 1)

        # Downsample x and y to (H // 2, W // 2, seq)
        x = x[:, ::2, ::2]
        y = y[:, ::2, ::2]
        return (
            np.array(x, dtype=np.float32),  # Input sequence
            np.array(y, dtype=np.int32)   # Target sequence
        )

def get_dataloaders(
    fname: str = "data/mnist_test_seq.npy",
    batch_size: int = 32,
    train_seq_length: int = 10
):
    train_idxs = (0, 8000)
    test_idxs = (8000, 10000)

    train_ds = MovingMNISTDataset(fname, train_idxs, train_seq_length)
    test_ds = MovingMNISTDataset(fname, test_idxs, train_seq_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=jnp_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=jnp_collate)

    return train_loader, test_loader
