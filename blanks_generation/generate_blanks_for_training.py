import struct
import random
import os

def generate_blank_images():
    """Generate 6000 blank 28x28 images with sparse random noise"""
    
    # IDX3 file format header
    # Magic number: 2051 (0x00000803)
    # Number of images: 6000 (0x00001770)
    # Number of rows: 28 (0x0000001C)
    # Number of columns: 28 (0x0000001C)
    
    with open('blank-train-images.idx3-ubyte', 'wb') as f:
        # Write header
        f.write(struct.pack('>I', 2051))    # Magic number (big-endian)
        f.write(struct.pack('>I', 6000))    # Number of images
        f.write(struct.pack('>I', 28))      # Number of rows
        f.write(struct.pack('>I', 28))      # Number of columns
        
        # Generate 6000 images
        for img_idx in range(6000):
            # Create 28x28 image (784 pixels)
            image = [0] * 784
            
            # 600 images are completely blank (all zeros)
            if img_idx < 600:
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
    
    print(f"Generated blank-train-images.idx3-ubyte with 6000 images")

def generate_blank_labels():
    """Generate 6000 labels, all set to 1 (representing space character)"""
    
    # IDX1 file format header
    # Magic number: 2049 (0x00000801)
    # Number of labels: 6000 (0x00001770)
    
    with open('blank-train-labels.idx1-ubyte', 'wb') as f:
        # Write header
        f.write(struct.pack('>I', 2049))    # Magic number (big-endian)
        f.write(struct.pack('>I', 6000))    # Number of labels
        
        # Write 6000 labels, all set to 1
        for _ in range(6000):
            f.write(struct.pack('B', 1))    # Label value 1 (unsigned byte)
    
    print(f"Generated blank-train-labels.idx1-ubyte with 6000 labels")

def generate_mapping_file():
    """Generate the mapping file"""
    with open('blank-mapping.txt', 'w') as f:
        f.write('1    32\n')  # Label 1 maps to ASCII 32 (space character)
    
    print(f"Generated blank-mapping.txt")

def verify_files():
    """Verify the generated files have correct sizes"""
    
    # Check images file
    img_expected_size = 16 + (6000 * 28 * 28)  # Header + image data
    img_actual_size = os.path.getsize('blank-train-images.idx3-ubyte')
    print(f"Images file size: {img_actual_size} bytes (expected: {img_expected_size})")
    
    # Check labels file  
    lbl_expected_size = 8 + 6000  # Header + label data
    lbl_actual_size = os.path.getsize('blank-train-labels.idx1-ubyte')
    print(f"Labels file size: {lbl_actual_size} bytes (expected: {lbl_expected_size})")
    
    # Check mapping file
    map_actual_size = os.path.getsize('blank-mapping.txt')
    print(f"Mapping file size: {map_actual_size} bytes")

if __name__ == "__main__":
    print("Generating blank MNIST-style dataset files...")
    
    # Set random seed for reproducible results (optional)
    random.seed(42)
    
    # Generate all three files
    generate_blank_images()
    generate_blank_labels() 
    generate_mapping_file()
    
    # Verify file sizes
    print("\nFile verification:")
    verify_files()
    
    print("\nDone! Generated files:")
    print("- blank-train-images.idx3-ubyte (IDX3 format, 6000 28x28 images)")
    print("- blank-train-labels.idx1-ubyte (IDX1 format, 6000 labels)")
    print("- blank-mapping.txt (label to ASCII mapping)")