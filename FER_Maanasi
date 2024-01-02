
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

# Define Emotion labels
idx_to_label = ["angry", "disgust", 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Path to the dataset
train_set = "D:\\MAIN DOCUMENT FOLDER\\Engineering\\SEM5\\MP - Mini Project\\archive\\train"
test_set = "D:\\MAIN DOCUMENT FOLDER\\Engineering\\SEM5\\MP - Mini Project\\archive\\test"

# Function to balance class distribution by oversampling
def balance_dataset_oversampling(path, label_to_freq, idx_to_label):
    samples = []
    max_samples = 0

    for label in idx_to_label:
        files = os.listdir(os.path.join(path, label))
        label_idx = idx_to_label.index(label)
        label_to_freq[label] = len(files)
        samples.extend([(os.path.join(path, label, file), label_idx) for file in files])
        max_samples = max(max_samples, label_to_freq[label])

    balanced_data = []
    for label in idx_to_label:
        label_samples = [(sample, label_idx) for sample, label_idx in samples if label_idx == idx_to_label.index(label)]
        oversampled_samples = label_samples * (max_samples // len(label_samples))
        np.random.shuffle(oversampled_samples)
        balanced_data.extend(oversampled_samples[:max_samples])

    return balanced_data


# Balance the train and test datasets using oversampling
train_data = balance_dataset_oversampling(train_set, {label: 0 for label in idx_to_label}, idx_to_label)
test_data = balance_dataset_oversampling(test_set, {label: 0 for label in idx_to_label}, idx_to_label)

# Define transformations for image data
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((20, 20)),
    transforms.ToTensor()
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


# Create train and test datasets
train_dataset = CustomDataset(train_data, transform=transform)
test_dataset = CustomDataset(test_data, transform=transform)

# Define DataLoaders
batch_size = 64
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the input to the fully connected layer

        self.flatten = nn.Flatten()

        self.fc_input_size = 128 * 2 * 2  # Update this value based on the shape after convolutions

        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.act4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, len(idx_to_label))

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))

        # Flatten the output before passing it to the fully connected layers
        x = self.flatten(x)

        # Ensure that the flattened shape matches the input size for the first fully connected layer
        x = self.dropout(self.act4(self.fc1(x)))
        x = self.fc2(x)
        return x


# Initialize model, criterion, optimizer, etc.
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
writer = SummaryWriter()

# Training loop
n_epochs = 1
models_dir = './models/'
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

    # Plot and save confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=idx_to_label, yticklabels=idx_to_label)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    figures_dir = './figures/'
    os.makedirs(figures_dir, exist_ok=True)

    confusion_figure_path = f"{figures_dir}confusion_matrix_epoch_{epoch + 1}.png"
    plt.savefig(confusion_figure_path)
    plt.close()

    # Save model checkpoint (if needed)
    model_path = f"{models_dir}model_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), model_path)

    # Print epoch statistics
    print(f"Epoch [{epoch + 1}/{n_epochs}], "
          f"Train Loss: {average_train_loss:.4f}, "
          f"Test Loss: {average_test_loss:.4f}, "
          f"Test Accuracy: {accuracy * 100:.2f}%")

    # Display confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=idx_to_label, yticklabels=idx_to_label)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()  # This will display the confusion matrix figure during training

writer.close()
