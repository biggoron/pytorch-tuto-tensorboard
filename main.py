import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from logger import Logger

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


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet().to(device)

logger = Logger('./logs')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

data_iter = iter(data_loader)
iter_per_epoch = len(data_loader)
total_step = 50000

for step in range(total_step):
    if (step+1) % iter_per_epoch == 0:
        data_iter = iter(data_loader)
    images, labels = next(data_iter)
    images, labels = images.view(images.size(0), -1).to(device), labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, argmax = torch.max(outputs, 1)
    accuracy = (labels == argmax.squeeze()).float().mean()

    if (step+1) % 100 == 0:
        print(f'Step [{step+1}/{total_step}], Loss: {loss.item()}, Acc: {accuracy.item()}')

    info = { 'loss': loss.items(), 'accuracy': accuracy.items() }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, step+1)

    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), step+1)
        logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step+1)

    info = { 'images': images.view(-1, 28, 28)[:10].cpu().numpy() }

    for tag, images in info.items():
        logger.image_summary(tag, images, step+1)
