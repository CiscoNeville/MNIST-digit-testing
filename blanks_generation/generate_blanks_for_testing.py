#! /usr/bin/python3

import struct
import random
import os

def generate_blank_test_images():
    """Generate 24000 blank 28x28 images with sparse random noise for testing"""
    
    # IDX3 file format header
    # Magic number: 2051 (0x00000803)
    # Number of images: 24000
    # Number of rows: 28 (0x0000001C)
    # Number of columns: 28 (0x0000001C)
    
    with open('blank-test-images.idx3-ubyte', 'wb') as f:
        # Write header
        f.write(struct.pack('>I', 2051))    # Magic number (big-endian)
        f.write(struct.pack('>I', 24000))   # Number of images
        f.write(struct.pack('>I', 28))      # Number of rows
        f.write(struct.pack('>I', 28))      # Number of columns
        
        # Generate 24000 images
        for img_idx in range(24000):
            # Create 28x28 image (784 pixels)
            image = [0] * 784
            
            # 2400 images are completely blank (all zeros) - 10% of total
            if img_idx < 2400:
                # Already all zeros, nothing to do
                pass
            else:
                # Add much more noise - 3-5% of pixels should be non-blank
                # 784 pixels * 0.03 to 0.05 = ~23 to 39 noise pixels
                num_noise_pixels = random.randint(23, 39)  # 3-5% of 784 pixels
                
                for _ in range(num_noise_pixels):
                    pixel_pos = random.randint(0, 783)
                    
                    # Generate noise centered at 0x1a with std dev 0x0a
                    # Using normal distribution, clamped to 1-255 range
                    pixel_value = random.gauss(0x1a, 0x0a)  # mean=26, std_dev=10
                    pixel_value = max(1, min(255, int(pixel_value)))  # Clamp to valid range
                    
                    image[pixel_pos] = pixel_value
            
            # Write image data
            f.write(bytes(image))
    
    print(f"Generated blank-test-images.idx3-ubyte with 24000 images")

def generate_blank_test_labels():
    """Generate 24000 labels, all set to 1 (representing space character)"""
    
    # IDX1 file format header
    # Magic number: 2049 (0x00000801)
    # Number of labels: 24000
    
    with open('blank-test-labels.idx1-ubyte', 'wb') as f:
        # Write header
        f.write(struct.pack('>I', 2049))    # Magic number (big-endian)
        f.write(struct.pack('>I', 24000))   # Number of labels
        
        # Write 24000 labels, all set to 1
        for _ in range(24000):
            f.write(struct.pack('B', 1))    # Label value 1 (unsigned byte)
    
    print(f"Generated blank-test-labels.idx1-ubyte with 24000 labels")

def verify_files():
    """Verify the generated files have correct sizes"""
    
    # Check images file
    img_expected_size = 16 + (24000 * 28 * 28)  # Header + image data
    img_actual_size = os.path.getsize('blank-test-images.idx3-ubyte')
    print(f"Images file size: {img_actual_size} bytes (expected: {img_expected_size})")
    
    # Check labels file  
    lbl_expected_size = 8 + 24000  # Header + label data
    lbl_actual_size = os.path.getsize('blank-test-labels.idx1-ubyte')
    print(f"Labels file size: {lbl_actual_size} bytes (expected: {lbl_expected_size})")

if __name__ == "__main__":
    print("Generating blank test dataset files...")
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # Generate both files
    generate_blank_test_images()
    generate_blank_test_labels() 
    
    # Verify file sizes
    print("\nFile verification:")
    verify_files()
    
    print("\nDone! Generated files:")
    print("- blank-test-images.idx3-ubyte (IDX3 format, 24000 28x28 images)")
    print("- blank-test-labels.idx1-ubyte (IDX1 format, 24000 labels)")