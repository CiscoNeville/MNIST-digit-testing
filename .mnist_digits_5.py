#! /usr/bin/python3

# train MNIST 0-9 dataset with pytorch
# Images loaded from file in IDX format
# Uses Apple Metal GPU if available
# Allows customizing training samples, test samples, and epochs via CLI arguments
# Can inspect specific weight connections during training

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
import re

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train MNIST classifier with custom settings')
parser.add_argument('--train_samples', type=int, default=60000, help='Number of training samples to use (max 60000)')
parser.add_argument('--test_samples', type=int, default=10000, help='Number of test samples to use (max 10000)')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--no_gpu', action='store_true', help='Disable GPU acceleration')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--train_seed', type=int, default=42, help='Random seed for training data selection and model initialization')
parser.add_argument('--test_seed', type=int, default=42, help='Random seed for test data selection')
parser.add_argument('--inspect', type=str, help='Inspect a specific weight connection: format "layer,neuron,output,interval" (e.g., "3,15,5,20")')
args = parser.parse_args()

# Parse inspection arguments if provided
inspect_params = None
if args.inspect:
    try:
        layer, neuron, output, interval = map(int, args.inspect.split(','))
        inspect_params = {
            'layer': layer,
            'neuron': neuron,
            'output': output,
            'interval': interval
        }
        print(f"Will inspect layer {layer}, neuron {neuron}, connection to output {output}, every {interval} batches")
    except:
        print("Error parsing --inspect argument. Format should be 'layer,neuron,output,interval'")
        inspect_params = None

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
        
# Define the model with named layers for easier inspection
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Create named layers instead of Sequential for better inspection access
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)  # Layer 1 (after input)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)  # Layer 2
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)   # Layer 3 (output)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x
    
    # Helper function to get a specific weight connection
    def get_specific_weight(self, layer, neuron, output):
        if layer == 1:
            # Layer 1 weights connect input to first hidden layer
            if neuron < self.fc1.weight.shape[0] and output < self.fc1.weight.shape[1]:
                return self.fc1.weight[neuron, output].item()
            else:
                raise ValueError(f"Indices out of range for layer 1 (max neuron: {self.fc1.weight.shape[0]-1}, max output: {self.fc1.weight.shape[1]-1})")
        elif layer == 2:
            # Layer 2 weights connect first hidden layer to second hidden layer
            if neuron < self.fc2.weight.shape[0] and output < self.fc2.weight.shape[1]:
                return self.fc2.weight[neuron, output].item()
            else:
                raise ValueError(f"Indices out of range for layer 2 (max neuron: {self.fc2.weight.shape[0]-1}, max output: {self.fc2.weight.shape[1]-1})")
        elif layer == 3:
            # Layer 3 weights connect second hidden layer to output layer
            if neuron < self.fc3.weight.shape[1] and output < self.fc3.weight.shape[0]:
                # Note the reversed indices for layer 3 since we're looking at connections FROM neuron TO output
                return self.fc3.weight[output, neuron].item()
            else:
                raise ValueError(f"Indices out of range for layer 3 (max neuron: {self.fc3.weight.shape[1]-1}, max output: {self.fc3.weight.shape[0]-1})")
        else:
            raise ValueError(f"Layer {layer} does not exist. Valid layers are 1, 2, and 3")

# Function to visualize a specific weight over time
def visualize_weight_evolution(weight_history):
    # Extract data for plotting
    batch_numbers = [0] + [data[0] for data in weight_history[1:]]
    weight_values = [data[1] for data in weight_history]
    
    # Calculate samples seen
    samples_seen = [batch_num * args.batch_size for batch_num in batch_numbers]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.plot(samples_seen, weight_values, 'o-', linewidth=2, markersize=8)
    
    # Add labels and title
    params = inspect_params
    plt.title(f"Evolution of Weight: Layer {params['layer']}, Neuron {params['neuron']} → Output {params['output']}", fontsize=14)
    plt.xlabel('Samples Seen', fontsize=12)
    plt.ylabel('Weight Value', fontsize=12)
    
    # Add grid and improve appearance
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Zero reference line
    
    # Add annotations for initial and final weight
    plt.annotate(f"Initial: {weight_values[0]:.4f}", 
                 xy=(samples_seen[0], weight_values[0]),
                 xytext=(samples_seen[0] + 0.05*max(samples_seen), weight_values[0]),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='green'))
    
    plt.annotate(f"Final: {weight_values[-1]:.4f}", 
                 xy=(samples_seen[-1], weight_values[-1]),
                 xytext=(samples_seen[-1] - 0.2*max(samples_seen), weight_values[-1]),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='blue'))
    
    plt.tight_layout()
    plt.show()

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

# Create training subset using train_seed
if train_samples < len(full_train_dataset):
    # Set seed specifically for training data selection
    random.seed(args.train_seed)
    train_indices = random.sample(range(len(full_train_dataset)), train_samples)
    train_dataset = Subset(full_train_dataset, train_indices)
else:
    train_dataset = full_train_dataset

# Create test subset using test_seed
if test_samples < len(full_test_dataset):
    # Set seed specifically for test data selection
    random.seed(args.test_seed)
    test_indices = random.sample(range(len(full_test_dataset)), test_samples)
    test_dataset = Subset(full_test_dataset, test_indices)
else:
    test_dataset = full_test_dataset

print(f"Using train_seed={args.train_seed} for training data selection")
print(f"Using test_seed={args.test_seed} for test data selection")
print(f"Using {train_samples} training samples and {test_samples} test samples")

# Set seeds for model initialization and training process using train_seed
random.seed(args.train_seed)
torch.manual_seed(args.train_seed)
np.random.seed(args.train_seed)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Initialize model, loss, and optimizer
model = MNISTClassifier().to(device)  # Move model to GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Store weights for inspection if requested
weight_history = []
if inspect_params:
    try:
        # Record initial weights - batch 0
        initial_weight = model.get_specific_weight(
            inspect_params['layer'], 
            inspect_params['neuron'],
            inspect_params['output']
        )
        weight_history.append((0, initial_weight))  # Store as tuple: (batch_number, weight_value)
        print(f"Recorded initial weight: {initial_weight:.6f}")
    except ValueError as e:
        print(f"Error recording initial weight: {e}")
        inspect_params = None

# Training loop
start_time = time.time()
num_epochs = args.epochs
print(f"Starting training for {num_epochs} epochs")

total_batches = 0
samples_seen = 0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    epoch_samples_processed = 0
    
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
        batch_size = images.size(0)
        running_loss += loss.item()
        epoch_samples_processed += batch_size
        total_batches += 1
        samples_seen += batch_size
        
        # Record weights at specified intervals if inspection is requested
        if inspect_params and total_batches % inspect_params['interval'] == 0:
            try:
                current_weight = model.get_specific_weight(
                    inspect_params['layer'], 
                    inspect_params['neuron'],
                    inspect_params['output']
                )
                weight_history.append((total_batches, current_weight))
                print(f"Batch {total_batches} (samples seen: {samples_seen}): weight = {current_weight:.6f}")
            except ValueError as e:
                print(f"Error recording weight at batch {total_batches}: {e}")
        
        if (i+1) % 50 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {i+1}, Samples: {epoch_samples_processed}/{train_samples}, Loss: {running_loss/50:.4f}')
            running_loss = 0.0

# Record final weight if it wasn't already recorded in the last interval
if inspect_params:
    try:
        # Only record if the last recorded batch isn't the final batch
        if weight_history[-1][0] != total_batches:
            final_weight = model.get_specific_weight(
                inspect_params['layer'], 
                inspect_params['neuron'],
                inspect_params['output']
            )
            weight_history.append((total_batches, final_weight))
            print(f"Recorded final weight: {final_weight:.6f}")
    except ValueError as e:
        print(f"Error recording final weight: {e}")

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

# Visualize weight evolution if inspection was requested
if inspect_params and weight_history:
    visualize_weight_evolution(weight_history)

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

# Show predictions for all test images (or firstr 5 if more than 5)
num_to_show = min(5, test_samples)
indices_to_show = list(range(num_to_show))  # Show first N test images in order
show_predictions(model, test_dataset, indices_to_show, device)
