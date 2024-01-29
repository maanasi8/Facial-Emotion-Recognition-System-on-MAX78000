from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
import torch.nn.functional as F

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

# Load test dataset without oversampling
for label in idx_to_label:
    files = os.listdir(os.path.join(test_set, label))
    label_idx = idx_to_label.index(label)
    label_to_freq[label] = len(files)
    samples.extend([(os.path.join(test_set, label, file), label_idx) for file in files])

# Display total number of images in the test dataset
print(f"Total number of images in the test dataset: {len(samples)}")

label_to_freq_unbalanced = {label: 0 for label in idx_to_label}
for sample in samples:
    label_to_freq_unbalanced[idx_to_label[sample[1]]] += 1

print("Unbalanced Class Distribution:")
print(label_to_freq_unbalanced)

plt.bar(idx_to_label, list(label_to_freq_unbalanced.values()))
plt.title('Class Distribution in Test Dataset')
plt.xlabel('Emotions')
plt.ylabel('Number of Samples')
plt.show()

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.Resize((224, 224)),  # Resize image to (224, 224) for ResNet18
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image data
])

# Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, transform=None, train=True):
        self.data = data
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label


# Define the ImprovedCNN class before instantiating the model
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super(ImprovedCNN, self).__init__()

        # Load pre-trained ResNet50 model
        resnet50 = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet50.children())[:-1])

        # Add batch normalization
        self.batch_norm = nn.BatchNorm1d(2048)  # ResNet50 has 2048 features in its final layer

        # Add a custom fully connected layer
        self.fc = nn.Linear(2048, num_classes)
        
        # Add dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Check if input image has only one channel (grayscale)
        if x.size(1) == 1:
            # Expand the single channel to three channels
            x = x.expand(-1, 3, -1, -1)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.batch_norm(x)
        x = self.fc(x)
        x = self.dropout(x)  # Apply dropout after the fully connected layer
        # Apply softmax activation
        x = F.softmax(x, dim=1)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model
model = ImprovedCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

# Create a DataLoader for the training dataset
batch_size = 64  # You can adjust the batch size as needed
train_dataset = CustomDataset(data=train, transform=transform)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Create a DataLoader for the test dataset
test_dataset = CustomDataset(data=samples, transform=transform, train=False)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Training loop
# Training loop
n_epochs = 100  # Change this to the desired number of epochs
models_dir = '/kaggle/working/models'
os.makedirs(models_dir, exist_ok=True)

# Initialize empty lists to store loss, accuracy, precision, and recall values
train_loss_values = []
test_loss_values = []
train_accuracy_values = []
test_accuracy_values = []
test_precision_values = []
test_recall_values = []

train_loss_steps = list(range(1, n_epochs + 1))
test_loss_steps = list(range(1, n_epochs + 1))
train_accuracy_steps = list(range(1, n_epochs + 1))
test_accuracy_steps = list(range(1, n_epochs + 1))

for epoch in range(n_epochs):
    # Training steps
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    class_correct_train = [0] * len(idx_to_label)
    class_total_train = [0] * len(idx_to_label)

    for i, (images, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, predicted_train = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

        for c in range(len(idx_to_label)):
            class_total_train[c] += (labels == c).sum().item()
            class_correct_train[c] += (predicted_train == labels)[labels == c].sum().item()

    # Calculate average training loss and accuracy for the epoch
    average_train_loss = train_loss / len(trainloader)
    train_accuracy = correct_train / total_train
    train_accuracy_values.append(train_accuracy)

    class_train_accuracy = [class_correct_train[c] / class_total_train[c] for c in range(len(idx_to_label))]
    print(f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {average_train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")
    print("Class-wise Train Accuracy:")
    for c, label in enumerate(idx_to_label):
        print(f"{label}: {class_train_accuracy[c] * 100:.2f}%")

    # Evaluate on the test set
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    all_labels = []
    all_predictions = []
    class_correct_test = [0] * len(idx_to_label)
    class_total_test = [0] * len(idx_to_label)

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted_test = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted_test == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_test.cpu().numpy())

            for c in range(len(idx_to_label)):
                class_total_test[c] += (labels == c).sum().item()
                class_correct_test[c] += (predicted_test == labels)[labels == c].sum().item()

    # Calculate average test loss, accuracy, precision, and recall for the epoch
    average_test_loss = test_loss / len(testloader)
    test_accuracy = correct_test / total_test
    test_accuracy_values.append(test_accuracy)

    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_predictions, average='weighted')

    # Append loss, accuracy, precision, and recall values to respective lists
    test_loss_values.append(average_test_loss)
    test_precision_values.append(precision)
    test_recall_values.append(recall)

    class_test_accuracy = [class_correct_test[c] / class_total_test[c] for c in range(len(idx_to_label))]
    print(f"Test Loss: {average_test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")
    print("Class-wise Test Accuracy:")
    for c, label in enumerate(idx_to_label):
        print(f"{label}: {class_test_accuracy[c] * 100:.2f}%")
    print(f"Test Precision: {precision:.4f}, Test Recall: {recall:.4f}\n")
