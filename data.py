import torch
import numpy as np
from torchvision import datasets, transforms


class Dataset():
    def __init__(self, batch_size=64):
        data_path = "./dataset"

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


        test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    

    def train(self):
        return self.train_loader

    def test(self):
        return self.test_loader
