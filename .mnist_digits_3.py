#! /usr/bin/python3

# train MNIST 0-9 dataset with pytorch
# Images loaded from file in IDX format
# Uses Apple Metal GPU if available
# Allows customizing training samples, test samples, and epochs via CLI arguments
# Can inspect specific perceptron weights during training

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
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--inspect', type=str, help='Inspect a specific perceptron weight: format "layer,neuron,interval" (e.g., "2,50,100")')
args = parser.parse_args()

# Parse inspection arguments if provided
inspect_params = None
if args.inspect:
    try:
        layer, neuron, interval = map(int, args.inspect.split(','))
        inspect_params = {
            'layer': layer,
            'neuron': neuron,
            'interval': interval
        }
        print(f"Will inspect layer {layer}, neuron {neuron}, every {interval} batches")
    except:
        print("Error parsing --inspect argument. Format should be 'layer,neuron,interval'")
        inspect_params = None

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
    
    # Helper function to get weights for a specific perceptron
    def get_perceptron_weights(self, layer, neuron):
        if layer == 1:
            # Layer 1 weights connect input to first hidden layer
            if neuron < self.fc1.weight.shape[0]:
                return self.fc1.weight[neuron].detach().cpu().numpy()
            else:
                raise ValueError(f"Neuron index {neuron} out of range for layer 1 (max {self.fc1.weight.shape[0]-1})")
        elif layer == 2:
            # Layer 2 weights connect first hidden layer to second hidden layer
            if neuron < self.fc2.weight.shape[0]:
                return self.fc2.weight[neuron].detach().cpu().numpy()
            else:
                raise ValueError(f"Neuron index {neuron} out of range for layer 2 (max {self.fc2.weight.shape[0]-1})")
        elif layer == 3:
            # Layer 3 weights connect second hidden layer to output layer
            if neuron < self.fc3.weight.shape[0]:
                return self.fc3.weight[neuron].detach().cpu().numpy()
            else:
                raise ValueError(f"Neuron index {neuron} out of range for layer 3 (max {self.fc3.weight.shape[0]-1})")
        else:
            raise ValueError(f"Layer {layer} does not exist. Valid layers are 1, 2, and 3")

# Function to visualize perceptron weights
def visualize_weights(weight_history, layer, neuron):
    # For plotting the weights in a better way
    fig, axes = plt.subplots(len(weight_history), 1, figsize=(10, 2.5 * len(weight_history)))
    if len(weight_history) == 1:
        axes = [axes]  # Make axes iterable for single history item
    
    # Get min and max across all weights for consistent scaling
    all_weights = np.concatenate([weights.flatten() for _, weights in weight_history])
    weight_min, weight_max = all_weights.min(), all_weights.max()
    abs_max = max(abs(weight_min), abs(weight_max))
    
    # Create a colormap for better visibility
    cmap = plt.cm.RdBu_r
    
    for i, (stage, weights) in enumerate(weight_history):
        # For line plots instead of image plots
        x = np.arange(len(weights))
        
        # Plot the weights as a bar chart
        bars = axes[i].bar(x, weights, width=0.8, color=cmap((weights+abs_max)/(2*abs_max)))
        
        # Set titles and labels
        axes[i].set_title(f"Layer {layer}, Neuron {neuron} - {stage}")
        axes[i].set_xlabel("Input Connection")
        axes[i].set_ylabel("Weight Value")
        
        # Set y-axis limits to be symmetric and consistent across plots
        axes[i].set_ylim(-abs_max*1.1, abs_max*1.1)
        
        # Add a horizontal line at zero
        axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-abs_max, abs_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes[i])
        cbar.set_label('Weight Value')
    
    plt.tight_layout()
    plt.suptitle(f"Weight Evolution for Layer {layer}, Neuron {neuron}", 
                 fontsize=16, y=1.0 + 0.05)
    plt.subplots_adjust(top=0.9)
    plt.show()

# Alternative visualization using a heatmap for multiple neurons - available but not used
def visualize_weights_heatmap(weight_history, layer, neuron):
    n_stages = len(weight_history)
    stage_names = [stage for stage, _ in weight_history]
    weights_data = [weights for _, weights in weight_history]
    
    # Create a figure with consistent size
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Combine weights into a 2D array for heatmap (stages × weights)
    combined_weights = np.vstack(weights_data)
    
    # Create heatmap
    im = ax.imshow(combined_weights, aspect='auto', cmap='RdBu_r')
    
    # Configure axes
    ax.set_yticks(np.arange(n_stages))
    ax.set_yticklabels(stage_names)
    ax.set_xlabel('Input Connection')
    ax.set_ylabel('Training Stage')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Weight Value')
    
    # Add title
    plt.title(f"Weight Evolution for Layer {layer}, Neuron {neuron}")
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

# Store weights for inspection if requested
weight_history = []
if inspect_params:
    try:
        # Record initial weights
        initial_weights = model.get_perceptron_weights(
            inspect_params['layer'], 
            inspect_params['neuron']
        )
        weight_history.append(("Initial", initial_weights))
    except ValueError as e:
        print(f"Error recording initial weights: {e}")
        inspect_params = None

# Training loop
start_time = time.time()
num_epochs = args.epochs
print(f"Starting training for {num_epochs} epochs")

total_batches = 0
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
        total_batches += 1
        
        # Record weights at specified intervals if inspection is requested
        if inspect_params and total_batches % inspect_params['interval'] == 0:
            try:
                current_weights = model.get_perceptron_weights(
                    inspect_params['layer'], 
                    inspect_params['neuron']
                )
                weight_history.append((f"Batch {total_batches}", current_weights))
                print(f"Recorded weights at batch {total_batches}")
            except ValueError as e:
                print(f"Error recording weights at batch {total_batches}: {e}")
        
        if (i+1) % 50 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {i+1}, Samples: {samples_processed}/{train_samples}, Loss: {running_loss/50:.4f}')
            running_loss = 0.0

# Record final weights if inspection is requested
if inspect_params:
    try:
        final_weights = model.get_perceptron_weights(
            inspect_params['layer'], 
            inspect_params['neuron']
        )
        weight_history.append(("Final", final_weights))
    except ValueError as e:
        print(f"Error recording final weights: {e}")

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

# Visualize weights if inspection was requested
if inspect_params and weight_history:
    visualize_weights(
        weight_history, 
        inspect_params['layer'], 
        inspect_params['neuron']
    )

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