import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DataSetLoader:
    def __init__(self, cfg):
        """
        Args:
        data_path: Path to store the MNIST dataset.
        batch_size: Batch size for dataloaders.

        Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test data.
        """
        self.data_path = cfg['data_path']
        self.batch_size = cfg['batch_size']
        os.makedirs(self.data_path, exist_ok=True)


    def load_data(self):
        trasnsofrm = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(root=self.data_path, train=True, download=True, transform=trasnsofrm)
        test_dataset = datasets.MNIST(root=self.data_path, train=False, download=True, transform=trasnsofrm)

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataset, test_dataset, test_loader