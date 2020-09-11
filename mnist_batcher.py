import torch
from torchvision import datasets, transforms


class Dataset:
    def __init__(self, _batch_size):
        super(Dataset, self).__init__()

        dataset_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_dataset = datasets.MNIST(
            "./data/mnist", train=True, download=True, transform=dataset_transform
        )
        test_dataset = datasets.MNIST(
            "./data/mnist", train=False, download=True, transform=dataset_transform
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=_batch_size, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=_batch_size, shuffle=False
        )
