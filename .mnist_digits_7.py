#! /usr/bin/python3

# train MNIST 0-9 dataset with pytorch
# Images loaded from file in IDX format
# Uses Apple Metal GPU if available
# Allows customizing training samples, test samples, and epochs via CLI arguments
# Can inspect specific weight connections during training
# Supports training on digits (0-9) and/or letters (A-Z, a-z) as "NaN" class

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import argparse
import random
import time
import re
from collections import defaultdict

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train MNIST classifier with custom settings')
parser.add_argument('--train_samples', type=int, default=60000, help='Number of training samples to use total')
parser.add_argument('--test_samples', type=int, default=10000, help='Number of test samples to use (max 10000)')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--no_gpu', action='store_true', help='Disable GPU acceleration')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--train_seed', type=int, default=42, help='Random seed for training data selection and model initialization')
parser.add_argument('--test_seed', type=int, default=42, help='Random seed for test data selection')
parser.add_argument('--train_classes', type=str, required=True, help='Classes to train on (comma-separated): d,nan,w. Must include d. Example: d,nan')
parser.add_argument('--inspect', type=str, help='Inspect a specific weight connection: format "layer,neuron,output,interval" (e.g., "3,15,5,20")')
args = parser.parse_args()

# Parse and validate train_classes argument
def parse_train_classes(classes_str):
    classes = set(classes_str.split(','))
    
    if 'd' not in classes:
        raise ValueError("Must include 'd' in train_classes")
    
    valid_classes = {'d', 'nan', 'w'}
    invalid = classes - valid_classes
    if invalid:
        raise ValueError(f"Invalid classes: {invalid}")
    
    return classes

try:
    train_classes = parse_train_classes(args.train_classes)
    print(f"Training on classes: {sorted(train_classes)}")
except ValueError as e:
    print(f"Error parsing --train_classes: {e}")
    exit(1)

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

# Custom Dataset class for MNIST digits
class MNISTDataset(Dataset):
    def __init__(self, images_file, labels_file, transform=None):
        self.images = read_idx(images_file)
        self.labels = read_idx(labels_file)
        self.transform = transform
        self.class_type = 'digit'  # For tracking dataset type
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        # Add channel dimension and convert to float
        image = image.reshape(1, 28, 28).astype(np.float32) / 255.0
        label = int(self.labels[idx])  # Keep original digit labels (0-9)
        
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

# Custom Dataset class for EMNIST letters (treated as "NaN" class)
class EMNISTLettersDataset(Dataset):
    def __init__(self, images_file, labels_file, transform=None):
        self.images = read_idx(images_file)
        self.labels = read_idx(labels_file)
        self.transform = transform
        self.class_type = 'letter'  # For tracking dataset type
        
        # EMNIST letters mapping: 1-26 = A-Z, 27-52 = a-z
        # We want to exclude: O(15), o(41), W(23), w(49)
        excluded_labels = {15, 23, 41, 49}  # O, W, o, w
        
        # Filter out excluded characters
        valid_indices = []
        for i, label in enumerate(self.labels):
            if label not in excluded_labels:
                valid_indices.append(i)
        
        # Keep only valid samples
        self.valid_indices = valid_indices
        print(f"EMNIST letters: {len(self.images)} total, {len(valid_indices)} after excluding O,o,W,w")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Map to actual index in the filtered dataset
        actual_idx = self.valid_indices[idx]
        image = self.images[actual_idx]
        
        # Add channel dimension and convert to float
        image = image.reshape(1, 28, 28).astype(np.float32) / 255.0
        label = 10  # All letters are class 10 (NaN)
        
        if self.transform:
            image = self.transform(image)
            
        return torch.tensor(image), torch.tensor(label)
    
    # Helper function to visualize an image
    def show_image(self, idx):
        actual_idx = self.valid_indices[idx]
        plt.figure(figsize=(3, 3))
        plt.imshow(self.images[actual_idx], cmap='gray')
        
        # Convert label to letter for display
        original_label = self.labels[actual_idx]
        if 1 <= original_label <= 26:
            letter = chr(ord('A') + original_label - 1)
        elif 27 <= original_label <= 52:
            letter = chr(ord('a') + original_label - 27)
        else:
            letter = f"label_{original_label}"
        
        plt.title(f"Letter: {letter} (NaN)")
        plt.axis('off')
        plt.show()

# Placeholder Dataset class for whitespace (future use)
class WhitespaceDataset(Dataset):
    def __init__(self, transform=None):
        # Placeholder - empty dataset for now
        self.images = np.array([])
        self.transform = transform
        self.class_type = 'whitespace'
    
    def __len__(self):
        return 0  # No whitespace data yet
    
    def __getitem__(self, idx):
        # Placeholder implementation
        raise IndexError("Whitespace dataset not implemented yet")

# Custom Dataset class that proportionally samples from multiple datasets
class ProportionalDataset(Dataset):
    def __init__(self, datasets_with_proportions, total_samples, seed=42):
        """
        datasets_with_proportions: list of (dataset, proportion) tuples
        total_samples: total number of samples to generate
        """
        self.datasets = []
        self.proportions = []
        self.samples_per_dataset = []
        self.dataset_indices = []
        
        # Set up random state for reproducible sampling
        self.rng = np.random.RandomState(seed)
        
        # Calculate total available samples
        total_available = sum(len(dataset) for dataset, _ in datasets_with_proportions if len(dataset) > 0)
        
        # Check if we're requesting too many samples
        if total_samples > total_available:
            raise ValueError(f"Requested {total_samples} samples but only {total_available} available across all datasets")
        
        # Calculate samples per dataset
        total_proportion = sum(prop for _, prop in datasets_with_proportions)
        for dataset, proportion in datasets_with_proportions:
            if len(dataset) == 0:  # Skip empty datasets
                continue
            self.datasets.append(dataset)
            self.proportions.append(proportion / total_proportion)
            samples_needed = int(total_samples * proportion / total_proportion)
            
            # Check if this dataset has enough samples
            if samples_needed > len(dataset):
                raise ValueError(f"Dataset '{dataset.class_type}' needs {samples_needed} samples but only has {len(dataset)} available")
            
            self.samples_per_dataset.append(samples_needed)
            
            # Generate indices for this dataset (no replacement needed now)
            indices = self.rng.choice(len(dataset), samples_needed, replace=False)
            self.dataset_indices.append(indices)
        
        # Create a mapping from global index to (dataset_idx, local_idx)
        self.index_mapping = []
        for dataset_idx, indices in enumerate(self.dataset_indices):
            for local_idx in indices:
                self.index_mapping.append((dataset_idx, local_idx))
        
        # Shuffle the mapping to interleave samples from different datasets
        self.rng.shuffle(self.index_mapping)
        
        print(f"Proportional dataset created:")
        for i, (dataset, samples) in enumerate(zip(self.datasets, self.samples_per_dataset)):
            print(f"  {dataset.class_type}: {samples} samples ({samples/len(self.index_mapping)*100:.1f}%)")
    
    def __len__(self):
        return len(self.index_mapping)
    
    def __getitem__(self, idx):
        dataset_idx, local_idx = self.index_mapping[idx]
        return self.datasets[dataset_idx][local_idx]
        
# Define the model with named layers for easier inspection
class MNISTClassifier(nn.Module):
    def __init__(self, num_classes=11):  # 0-9 digits + 1 NaN class
        super().__init__()
        # Create named layers instead of Sequential for better inspection access
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)  # Layer 1 (after input)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)  # Layer 2
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)   # Layer 3 (output)
        self.num_classes = num_classes
    
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
train_images_NaN_path = os.path.join(data_dir, "emnist-letters-train-images.idx3-ubyte")  # Not a Number
train_labels_NaN_path = os.path.join(data_dir, "emnist-letters-train-labels.idx1-ubyte")  # Not a Number labels

# Create datasets based on train_classes
datasets_for_training = []

# Always include digits (required)
full_mnist_dataset = MNISTDataset(train_images_path, train_labels_path)
datasets_for_training.append((full_mnist_dataset, 1.0))  # Start with proportion 1.0

# Add EMNIST letters if requested
if 'nan' in train_classes:
    if os.path.exists(train_images_NaN_path) and os.path.exists(train_labels_NaN_path):
        full_emnist_dataset = EMNISTLettersDataset(train_images_NaN_path, train_labels_NaN_path)
        # Calculate proportion: EMNIST has ~145,600 samples, MNIST has 60,000
        # We want roughly equal representation, so use 0.1 proportion for EMNIST
        datasets_for_training.append((full_emnist_dataset, 0.1))
        print(f"Loaded EMNIST letters dataset: {len(full_emnist_dataset)} samples")
    else:
        missing_files = []
        if not os.path.exists(train_images_NaN_path):
            missing_files.append("emnist-letters-train-images.idx3-ubyte")
        if not os.path.exists(train_labels_NaN_path):
            missing_files.append("emnist-letters-train-labels.idx1-ubyte")
        print(f"Warning: EMNIST files not found: {missing_files}")
        print("Continuing with digits only...")

# Add whitespace if requested (placeholder)
if 'w' in train_classes:
    whitespace_dataset = WhitespaceDataset()
    if len(whitespace_dataset) > 0:
        datasets_for_training.append((whitespace_dataset, 0.05))
    else:
        print("Warning: Whitespace dataset not implemented yet, skipping...")

# Create proportional training dataset
train_dataset = ProportionalDataset(datasets_for_training, args.train_samples, args.train_seed)

# Create test dataset (digits only for now)
full_test_dataset = MNISTDataset(test_images_path, test_labels_path)

# Validate and adjust test sample count
test_samples = min(args.test_samples, len(full_test_dataset))

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
print(f"Using {len(train_dataset)} training samples and {test_samples} test samples")

# Set seeds for model initialization and training process using train_seed
random.seed(args.train_seed)
torch.manual_seed(args.train_seed)
np.random.seed(args.train_seed)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Determine number of classes
num_classes = 10  # Start with digits 0-9
if 'nan' in train_classes:
    num_classes = 11  # Add NaN class
if 'w' in train_classes:
    num_classes = 12  # Add whitespace class (future)

# Initialize model, loss, and optimizer
model = MNISTClassifier(num_classes=num_classes).to(device)  # Move model to GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Model created with {num_classes} output classes")

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
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {i+1}, Samples: {epoch_samples_processed}/{len(train_dataset)}, Loss: {running_loss/50:.4f}')
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
print(f"Average time per sample: {training_time/(num_epochs * len(train_dataset)) * 1000:.2f} ms")

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    confusion_matrix = defaultdict(lambda: defaultdict(int))  # [true_class][predicted_class] = count
    
    for images, labels in test_loader:
        # Move data to device
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Track per-class accuracy and confusion matrix
        for i in range(labels.size(0)):
            true_label = labels[i].item()
            pred_label = predicted[i].item()
            class_total[true_label] += 1
            confusion_matrix[true_label][pred_label] += 1
            if pred_label == true_label:
                class_correct[true_label] += 1

    print(f'Overall accuracy on {test_samples} test images: {100 * correct / total:.2f}%')
    
    # Helper function to convert class ID to readable name
    def class_name(class_id):
        if class_id <= 9:
            return str(class_id)
        elif class_id == 10:
            return "NaN"
        elif class_id == 11:
            return "ws"
        else:
            return f"class_{class_id}"
    
    # Print detailed per-class breakdown
    print("\nDetailed breakdown:")
    for true_class in sorted(class_total.keys()):
        total_for_class = class_total[true_class]
        correct_for_class = class_correct[true_class]
        accuracy = 100 * correct_for_class / total_for_class if total_for_class > 0 else 0
        
        # Build incorrect predictions string
        incorrect_parts = []
        for pred_class, count in confusion_matrix[true_class].items():
            if pred_class != true_class and count > 0:
                incorrect_parts.append(f"{count} ({class_name(pred_class)})")
        
        incorrect_str = ", ".join(incorrect_parts) if incorrect_parts else "none"
        
        print(f'  Class {class_name(true_class)}: {accuracy:.1f}% ({correct_for_class}/{total_for_class}) correct, incorrect: {incorrect_str}')
    
    # Print confusion matrix if there are misclassifications
    if correct < total:
        print(f"\nConfusion Matrix (rows=true, cols=predicted):")
        
        # Get all classes that appear in predictions or true labels
        all_classes = set(class_total.keys())
        for true_class in confusion_matrix:
            all_classes.update(confusion_matrix[true_class].keys())
        all_classes = sorted(all_classes)
        
        # Print header
        header = "True\\Pred\n"
        for pred_class in all_classes:
            header += f"\t{class_name(pred_class)}"
        print(header)
        
        # Print rows
        for true_class in all_classes:
            if true_class in class_total:  # Only show classes that exist in test data
                row = f"{class_name(true_class)}"
                for pred_class in all_classes:
                    count = confusion_matrix[true_class][pred_class]
                    row += f"\t{count}"
                print(row)

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
            
            # Convert labels to readable format
            true_label = label.item()
            pred_label = predicted.item()
            
            true_str = str(true_label) if true_label <= 9 else ("NaN" if true_label == 10 else "ws")
            pred_str = str(pred_label) if pred_label <= 9 else ("NaN" if pred_label == 10 else "ws")
            
            # Get the actual image from the parent dataset
            if isinstance(dataset, Subset):
                img_to_show = parent_dataset.images[real_indices[i]]
            else:
                img_to_show = parent_dataset.images[idx]
            
            axes[i].imshow(img_to_show, cmap='gray')
            axes[i].set_title(f"True: {true_str}\nPred: {pred_str}")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Show predictions for all test images (or first 5 if more than 5)
num_to_show = min(5, test_samples)
indices_to_show = list(range(num_to_show))  # Show first N test images in order
show_predictions(model, test_dataset, indices_to_show, device)