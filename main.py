import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset
dataset = torchvision.datasets.MNIST(
    root='../../data',
    train=True,
    transform=transforms.ToTensor(),
    download=True)
data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=100,
    shuffle=True)

