from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils, datasets
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Emotion labels
idx_to_label = ["angry", "disgust", 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Path to the dataset
train_set = "/kaggle/input/ferdata/train"
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

# Display total number of images after balancing the class distribution in the train dataset
print(f"Total number of images after balancing in the train dataset: {len(train)}")

label_to_freq_balanced = {label: 0 for label in idx_to_label}
for sample in train:
    label_to_freq_balanced[idx_to_label[sample[1]]] += 1

print("Balanced Class Distribution:")
print(label_to_freq_balanced)

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
test_set = "/kaggle/input/ferdata/test"

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

# Display total number of images after balancing the class distribution in the test dataset
print(f"Total number of images after balancing in the test dataset: {len(test)}")

label_to_freq_balanced = {label: 0 for label in idx_to_label}
for sample in test:
    label_to_freq_balanced[idx_to_label[sample[1]]] += 1

print("Balanced Class Distribution:")
print(label_to_freq_balanced)

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

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.Resize((28, 28)),  # Resize image to (28, 28)
    transforms.ToTensor()  # Convert PIL image to tensor
])

# Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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

# class CNNModel(nn.Module):
#     def _init_(self):
#         super(CNNModel, self)._init_()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.act1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.act2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.act3 = nn.ReLU()
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Calculate the size of the input to the fully connected layer

#         self.flatten = nn.Flatten()

#         self.fc_input_size = 128 * 2 * 2  # Update this value based on the shape after convolutions

#         self.fc1 = nn.Linear(self.fc_input_size, 512)
#         self.act4 = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(512, len(idx_to_label))

#     def forward(self, x):
#         x = self.pool1(self.act1(self.conv1(x)))
#         x = self.pool2(self.act2(self.conv2(x)))
#         x = self.pool3(self.act3(self.conv3(x)))

#         # Flatten the output before passing it to the fully connected layers
#         x = self.flatten(x)

#         # Ensure that the flattened shape matches the input size for the first fully connected layer
#         x = self.dropout(self.act4(self.fc1(x)))
#         x = self.fc2(x)
#         return x


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):  # Assuming 7 classes for emotions
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # Adjust input size based on your dimensions
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x



# Initialize model, criterion, optimizer, etc.
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

writer = SummaryWriter()

# Training loop
n_epochs = 100
models_dir = '/kaggle/working/models'
os.makedirs(models_dir, exist_ok=True)

for epoch in range(n_epochs):
    # Training steps
    model.train()
    train_loss = 0.0
    for i, (images, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Calculate average training loss for the epoch
    average_train_loss = train_loss / len(trainloader)

    # Evaluate on the test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())

    # Calculate average test loss and accuracy for the epoch
    average_test_loss = test_loss / len(testloader)
    accuracy = correct / total

    # Calculate and log additional metrics like confusion matrix
    confusion = confusion_matrix(all_labels, all_predictions)

    # Log metrics to TensorBoard
    writer.add_scalar('Loss/Train', average_train_loss, epoch + 1)
    writer.add_scalar('Loss/Test', average_test_loss, epoch + 1)
    writer.add_scalar('Accuracy/Test', accuracy, epoch + 1)

    # Plot and display confusion matrix without saving
    plt.figure(figsize=(8, 8))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=idx_to_label, yticklabels=idx_to_label)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()  # Display the confusion matrix during training

    # Save model checkpoint (if needed)
    model_path = f"{models_dir}/model_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), model_path)

    # Print epoch statistics
    print(f"Epoch [{epoch + 1}/{n_epochs}], "
          f"Train Loss: {average_train_loss:.4f}, "
          f"Test Loss: {average_test_loss:.4f}, "
          f"Test Accuracy: {accuracy * 100:.2f}%")

writer.close()
