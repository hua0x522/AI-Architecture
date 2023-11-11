import torch
import torch.nn as nn
import torch.optim as optim

from data import Dataset
from model import LeNet

dataset = Dataset()
lenet = LeNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lenet.parameters(), lr=0.001)

num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lenet.to(device)

for epoch in range(num_epochs):
    train_loss = 0.0
    lenet.train()

    for images, labels in dataset.train():
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = lenet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(dataset.train()):.4f}")

lenet.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for images, labels in dataset.test():
        images = images.to(device)
        labels = labels.to(device)

        outputs = lenet(images)
        _, predicted = torch.max(outputs, 1)

        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

accuracy = 100.0 * total_correct / total_samples
print(f"Test Accuracy: {accuracy:.2f}%")
