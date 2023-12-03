# Mini-Project

## code:
from torchvision import transforms
import os
import numpy as np
import matplotlib.pyplot as plt

idx_to_label = ["angry", "disgust", 'fear', 'happy', 'neutral', 'sad', 'surprise']

dataset_path = "D:\\MAIN DOCUMENT FOLDER\\Engineering\\SEM5\\MP - Mini Project\\archive\\train"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

label_to_freq = {label: 0 for label in idx_to_label}
samples = []

for label in idx_to_label:
    files = os.listdir(os.path.join(dataset_path, label))
    label_idx = idx_to_label.index(label)

    label_to_freq[label] = len(files)
    samples.extend([(os.path.join(dataset_path, label, file), label_idx) for file in files])

max_samples = max(label_to_freq.values())
oversampled_samples = samples.copy()

for label in idx_to_label:
    label_count = label_to_freq[label]
    if label_count < max_samples:
        samples_to_add = max_samples - label_count
        label_indices = [idx for idx, sample in enumerate(samples) if sample[1] == idx_to_label.index(label)]
        selected_samples = np.random.choice(label_indices, samples_to_add, replace=True)

        for idx in selected_samples:
            oversampled_samples.append(samples[idx])

label_to_freq_balanced = {label: 0 for label in idx_to_label}
for sample in oversampled_samples:
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

## Results:

![alt text](![Figure_1](https://github.com/maanasi8/Mini-Project/assets/126388400/794ea0b6-570d-4a63-8e97-c8611c6bb674)
)
