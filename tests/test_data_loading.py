import unittest
from data_loading import get_dataloaders


class TestDataLoading(unittest.TestCase):
    def test_get_dataloaders(self):
        train_dl, test_dl = get_dataloaders()
        self.assertEqual(len(train_dl.dataset), 8000)
        self.assertEqual(len(test_dl.dataset), 2000)

    def test_data_shape(self):
        train_dl, test_dl = get_dataloaders(
            batch_size=16,
            train_seq_length=10
        )
        x1, y1 = next(iter(train_dl))
        x2, y2 = next(iter(test_dl))

        bs, H, W, seq_len = x1.shape
        print("Train data shape:", x1.shape)
        self.assertEqual(seq_len, 10)
        self.assertEqual(bs, 16)  # Check if batch size is correct

        bs, H, W, seq_len = x2.shape
        print("Test data shape:", x2.shape)
        self.assertEqual(seq_len, 10)
        self.assertEqual(bs, 16)  # Check if batch size is correct


if __name__ == "__main__":
    unittest.main()
