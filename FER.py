# Importing necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.utils import make_grid

# Emotion labels
idx_to_label = ["angry", "disgust", 'fear', 'happy', 'neutral', 'sad', 'surprise']
# Path to the dataset
train_set = "./train"

label_to_freq = {label: 0 for label in idx_to_label}
samples = []

# Code to balance the class distribution in the train dataset by oversampling
for label in idx_to_label:
    files = os.listdir(os.path.join(train_set, label))
    label_idx = idx_to_label.index(label)

    label_to_freq[label] = len(files)
    samples.extend([(os.path.join(train_set, label, file), label_idx) for file in files])

max_samples = max(label_to_freq.values())
train = samples.copy()

for label in idx_to_label:
    label_count = label_to_freq[label]
    if label_count < max_samples:
        samples_to_add = max_samples - label_count
        label_indices = [idx for idx, sample in enumerate(samples) if sample[1] == idx_to_label.index(label)]
        selected_samples = np.random.choice(label_indices, samples_to_add, replace=True)

        for idx in selected_samples:
            train.append(samples[idx])

label_to_freq_balanced = {label: 0 for label in idx_to_label}
for sample in train:
    label_to_freq_balanced[idx_to_label[sample[1]]] += 1

print("Balanced Class Distribution:")
print(label_to_freq_balanced)

# Displaying the balancing with the help of  histograms
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(idx_to_label, list(label_to_freq.values()))
plt.title('Class Distribution Before Balancing')
plt.xlabel('Emotions')
plt.ylabel('Number of Samples')

plt.subplot(1, 2, 2)
plt.bar(idx_to_label, list(label_to_freq_balanced.values()))
plt.title('Class Distribution After Balancing')
plt.xlabel('Emotions')
plt.ylabel('Number of Samples')

plt.tight_layout()
plt.show()

# Emotion labels
idx_to_label = ["angry", "disgust", 'fear', 'happy', 'neutral', 'sad', 'surprise']
# Path to the dataset
test_set = "./test"

label_to_freq = {label: 0 for label in idx_to_label}
samples = []

# Code to balance the class distribution in the test dataset by oversampling
for label in idx_to_label:
    files = os.listdir(os.path.join(test_set, label))
    label_idx = idx_to_label.index(label)

    label_to_freq[label] = len(files)
    samples.extend([(os.path.join(test_set, label, file), label_idx) for file in files])

max_samples = max(label_to_freq.values())
test = samples.copy()

for label in idx_to_label:
    label_count = label_to_freq[label]
    if label_count < max_samples:
        samples_to_add = max_samples - label_count
        label_indices = [idx for idx, sample in enumerate(samples) if sample[1] == idx_to_label.index(label)]
        selected_samples = np.random.choice(label_indices, samples_to_add, replace=True)

        for idx in selected_samples:
            test.append(samples[idx])

label_to_freq_balanced = {label: 0 for label in idx_to_label}
for sample in test:
    label_to_freq_balanced[idx_to_label[sample[1]]] += 1

print("Balanced Class Distribution:")
print(label_to_freq_balanced)

# Displaying the balancing with the help of  histograms
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(idx_to_label, list(label_to_freq.values()))
plt.title('Class Distribution Before Balancing')
plt.xlabel('Emotions')
plt.ylabel('Number of Samples')

plt.subplot(1, 2, 2)
plt.bar(idx_to_label, list(label_to_freq_balanced.values()))
plt.title('Class Distribution After Balancing')
plt.xlabel('Emotions')
plt.ylabel('Number of Samples')

plt.tight_layout()
plt.show()

# Define transformations for image data
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
     torchvision.transforms.Resize((48, 48)),  # Resize image to (48, 48)
     torchvision.transforms.ToTensor()])  # Convert PIL image to tensor

# Define custom dataset class
class CustomDataset(Dataset):
    # Initialize dataset with data and transformation
    def _init_(self, data, transform=None):
        self.data = data
        self.transform = transform
    # Return the length of the dataset
    def _len_(self):
        return len(self.data)
    # Get item at a particular index from the dataset
    def _getitem_(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label

# Define train and test datasets
train_dataset = CustomDataset(train, transform=transform)
test_dataset = CustomDataset(test, transform=transform)

# DataLoaders
batch_size = 64
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# CNN Model
class CNNModel(nn.Module):
    def _init_(self):
        super(CNNModel, self)._init_()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.act4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, len(idx_to_label))

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        x = self.flatten(x)
        x = self.dropout(self.act4(self.fc1(x)))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = CNNModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Calculate accuracy on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {running_loss / len(trainloader):.4f}, '
          f'Test Accuracy: {(100 * correct / total):.2f}%')

# Displaying a batch of images
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0), cmap='gray')
        break
        
show_batch(trainloader)  
