#!/usr/bin/env python3

import struct
import random
import os

def read_idx_header(filename):
    """Read IDX file header and return metadata"""
    with open(filename, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num_items = struct.unpack('>I', f.read(4))[0]
        
        # Determine data type from magic number
        data_type = magic & 0xFF
        
        if data_type == 0x03:  # Images (IDX3) - magic number 2051
            num_rows = struct.unpack('>I', f.read(4))[0]
            num_cols = struct.unpack('>I', f.read(4))[0]
            return magic, num_items, num_rows, num_cols
        elif data_type == 0x01:  # Labels (IDX1) - magic number 2049
            return magic, num_items, None, None
        else:
            raise ValueError(f"Unknown magic number: {magic} (data_type: {data_type})")

def read_all_images(filename):
    """Read all images from IDX3 file"""
    with open(filename, 'rb') as f:
        magic, num_images, num_rows, num_cols = read_idx_header(filename)
        
        # Skip header (already read)
        f.seek(16)
        
        # Read all image data
        image_size = num_rows * num_cols
        images = []
        for _ in range(num_images):
            image_data = f.read(image_size)
            images.append(image_data)
        
        return images, num_rows, num_cols

def read_all_labels(filename):
    """Read all labels from IDX1 file"""
    with open(filename, 'rb') as f:
        magic, num_labels, _, _ = read_idx_header(filename)
        
        # Skip header
        f.seek(8)
        
        # Read all label data
        labels = []
        for _ in range(num_labels):
            label = struct.unpack('B', f.read(1))[0]
            labels.append(label)
        
        return labels

def write_idx3_file(filename, images, num_rows, num_cols):
    """Write images to IDX3 file"""
    with open(filename, 'wb') as f:
        # Write header
        f.write(struct.pack('>I', 2051))           # Magic number for images
        f.write(struct.pack('>I', len(images)))    # Number of images
        f.write(struct.pack('>I', num_rows))       # Number of rows
        f.write(struct.pack('>I', num_cols))       # Number of columns
        
        # Write all images
        for image_data in images:
            f.write(image_data)

def write_idx1_file(filename, labels):
    """Write labels to IDX1 file"""
    with open(filename, 'wb') as f:
        # Write header
        f.write(struct.pack('>I', 2049))          # Magic number for labels
        f.write(struct.pack('>I', len(labels)))   # Number of labels
        
        # Write all labels
        for label in labels:
            f.write(struct.pack('B', label))

def split_emnist_letters(images_file, labels_file, train_size=80000, test_size=8434, seed=42):
    """
    Split EMNIST letters dataset into training and testing sets.
    
    Args:
        images_file: Path to input images file
        labels_file: Path to input labels file
        train_size: Number of samples for training (default: 80000)
        test_size: Number of samples for testing (default: 8434)
        seed: Random seed for reproducibility (default: 42)
    """
    
    print(f"Reading EMNIST letters dataset...")
    
    # Read all data
    images, num_rows, num_cols = read_all_images(images_file)
    labels = read_all_labels(labels_file)
    
    total_samples = len(images)
    print(f"Total samples found: {total_samples}")
    
    if len(images) != len(labels):
        raise ValueError(f"Mismatch: {len(images)} images but {len(labels)} labels")
    
    required_samples = train_size + test_size
    if total_samples < required_samples:
        raise ValueError(f"Not enough samples! Need {required_samples}, but only have {total_samples}")
    
    print(f"Splitting into {train_size} training and {test_size} testing samples...")
    
    # Create indices and shuffle
    random.seed(seed)
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    # Split indices
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]
    
    # Create training set
    train_images = [images[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    
    # Create testing set
    test_images = [images[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    # Write training files
    print(f"Writing emnist-letters-train-images.idx3-ubyte ({len(train_images)} samples)...")
    write_idx3_file('emnist-letters-train-images.idx3-ubyte', train_images, num_rows, num_cols)
    
    print(f"Writing emnist-letters-train-labels.idx1-ubyte ({len(train_labels)} labels)...")
    write_idx1_file('emnist-letters-train-labels.idx1-ubyte', train_labels)
    
    # Write testing files
    print(f"Writing emnist-letters-test-images.idx3-ubyte ({len(test_images)} samples)...")
    write_idx3_file('emnist-letters-test-images.idx3-ubyte', test_images, num_rows, num_cols)
    
    print(f"Writing emnist-letters-test-labels.idx1-ubyte ({len(test_labels)} labels)...")
    write_idx1_file('emnist-letters-test-labels.idx1-ubyte', test_labels)
    
    print("\nVerifying file sizes...")
    verify_split_files(train_size, test_size, num_rows, num_cols)
    
    print("\nDone! Generated files:")
    print(f"  Training: emnist-letters-train-images.idx3-ubyte ({train_size} images)")
    print(f"            emnist-letters-train-labels.idx1-ubyte ({train_size} labels)")
    print(f"  Testing:  emnist-letters-test-images.idx3-ubyte ({test_size} images)")
    print(f"            emnist-letters-test-labels.idx1-ubyte ({test_size} labels)")

def verify_split_files(train_size, test_size, num_rows, num_cols):
    """Verify the generated files have correct sizes"""
    
    # Training files
    train_img_expected = 16 + (train_size * num_rows * num_cols)
    train_img_actual = os.path.getsize('emnist-letters-train-images.idx3-ubyte')
    print(f"  Train images: {train_img_actual} bytes (expected: {train_img_expected})")
    
    train_lbl_expected = 8 + train_size
    train_lbl_actual = os.path.getsize('emnist-letters-train-labels.idx1-ubyte')
    print(f"  Train labels: {train_lbl_actual} bytes (expected: {train_lbl_expected})")
    
    # Testing files
    test_img_expected = 16 + (test_size * num_rows * num_cols)
    test_img_actual = os.path.getsize('emnist-letters-test-images.idx3-ubyte')
    print(f"  Test images:  {test_img_actual} bytes (expected: {test_img_expected})")
    
    test_lbl_expected = 8 + test_size
    test_lbl_actual = os.path.getsize('emnist-letters-test-labels.idx1-ubyte')
    print(f"  Test labels:  {test_lbl_actual} bytes (expected: {test_lbl_expected})")

if __name__ == "__main__":
    import sys
    
    # Default file names (adjust if your files have different names)
    images_file = 'emnist-letters-train-images.idx3-ubyte'
    labels_file = 'emnist-letters-train-labels.idx1-ubyte'
    
    # Check if files exist
    if not os.path.exists(images_file):
        print(f"Error: {images_file} not found!")
        print("Please ensure the EMNIST letters training files are in the current directory.")
        sys.exit(1)
    
    if not os.path.exists(labels_file):
        print(f"Error: {labels_file} not found!")
        print("Please ensure the EMNIST letters training files are in the current directory.")
        sys.exit(1)
    
    # Perform the split
    split_emnist_letters(images_file, labels_file, train_size=80000, test_size=8434)