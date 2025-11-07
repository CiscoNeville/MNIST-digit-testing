#! /usr/bin/python3

# train MNIST 0-9 dataset with pytorch
# Images loaded from file in IDX format
# Uses Apple Metal GPU if available
# Allows customizing training samples, test samples, and epochs via CLI arguments

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import argparse
import random
import time

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train MNIST classifier with custom settings')
parser.add_argument('--train_samples', type=int, default=60000, help='Number of training samples to use (max 60000)')
parser.add_argument('--test_samples', type=int, default=10000, help='Number of test samples to use (max 10000)')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--no_gpu', action='store_true', help='Disable GPU acceleration')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()

# Set random seed for reproducibility
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Function to read IDX files
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

# Custom Dataset class for MNIST
class MNISTDataset(Dataset):
    def __init__(self, images_file, labels_file, transform=None):
        self.images = read_idx(images_file)
        self.labels = read_idx(labels_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        # Add channel dimension and convert to float
        image = image.reshape(1, 28, 28).astype(np.float32) / 255.0
        label = int(self.labels[idx])
        
        if self.transform:
            image = self.transform(image)
            
        return torch.tensor(image), torch.tensor(label)
    
    # Helper function to visualize an image
    def show_image(self, idx):
        plt.figure(figsize=(3, 3))
        plt.imshow(self.images[idx], cmap='gray')
        plt.title(f"Label: {self.labels[idx]}")
        plt.axis('off')
        plt.show()
        
# Define the model
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),  # Flatten the 28x28 image
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

# Use Apple Metal (MPS) GPU if available and not disabled
if args.no_gpu:
    device = torch.device("cpu")
else:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Set paths
data_dir = "~/code/mnist_digits"
data_dir = os.path.expanduser(data_dir)  # Expand the ~ in the path

train_images_path = os.path.join(data_dir, "train-images.idx3-ubyte")
train_labels_path = os.path.join(data_dir, "train-labels.idx1-ubyte")
test_images_path = os.path.join(data_dir, "t10k-images.idx3-ubyte")
test_labels_path = os.path.join(data_dir, "t10k-labels.idx1-ubyte")

# Create full datasets
full_train_dataset = MNISTDataset(train_images_path, train_labels_path)
full_test_dataset = MNISTDataset(test_images_path, test_labels_path)

# Validate and adjust sample counts
train_samples = min(args.train_samples, len(full_train_dataset))
test_samples = min(args.test_samples, len(full_test_dataset))

# Create subsets of the data if needed
if train_samples < len(full_train_dataset):
    train_indices = random.sample(range(len(full_train_dataset)), train_samples)
    train_dataset = Subset(full_train_dataset, train_indices)
else:
    train_dataset = full_train_dataset

if test_samples < len(full_test_dataset):
    test_indices = random.sample(range(len(full_test_dataset)), test_samples)
    test_dataset = Subset(full_test_dataset, test_indices)
else:
    test_dataset = full_test_dataset

print(f"Using {train_samples} training samples and {test_samples} test samples")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Initialize model, loss, and optimizer
model = MNISTClassifier().to(device)  # Move model to GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
start_time = time.time()
num_epochs = args.epochs
print(f"Starting training for {num_epochs} epochs")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    samples_processed = 0
    
    for i, (images, labels) in enumerate(train_loader):
        # Move data to device
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update tracking variables
        running_loss += loss.item()
        samples_processed += images.size(0)
        
        if (i+1) % 50 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {i+1}, Samples: {samples_processed}/{train_samples}, Loss: {running_loss/50:.4f}')
            running_loss = 0.0

# Calculate training time
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")
print(f"Average time per epoch: {training_time/num_epochs:.2f} seconds")
print(f"Average time per sample: {training_time/(num_epochs * train_samples) * 1000:.2f} ms")

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        # Move data to device
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy on {test_samples} test images: {100 * correct / total:.2f}%')

# Function to visualize predictions
def show_predictions(model, dataset, indices, device):
    model.eval()
    
    # Handle both full dataset and subset
    if isinstance(dataset, Subset):
        parent_dataset = dataset.dataset
        real_indices = [dataset.indices[i] for i in indices]
    else:
        parent_dataset = dataset
        real_indices = indices
    
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 3))
    if len(indices) == 1:
        axes = [axes]  # Make iterable for single image case
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get image and label
            image, label = dataset[idx]
            
            # Move to device for prediction
            image = image.to(device).unsqueeze(0)
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            
            # Get the actual image from the parent dataset
            if isinstance(dataset, Subset):
                img_to_show = parent_dataset.images[real_indices[i]]
            else:
                img_to_show = parent_dataset.images[idx]
            
            axes[i].imshow(img_to_show, cmap='gray')
            axes[i].set_title(f"True: {label}\nPred: {predicted.item()}")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Show predictions for 5 random test images
if test_samples >= 5:
    random_indices = random.sample(range(len(test_dataset)), 5)
    show_predictions(model, test_dataset, random_indices, device)
else:
    random_indices = random.sample(range(len(test_dataset)), min(5, test_samples))
    show_predictions(model, test_dataset, random_indices, device)