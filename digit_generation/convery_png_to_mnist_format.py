#!/usr/bin/env python3
"""
convert_png_to_mnist_format.py
Advanced MNIST converter with brightness control and matrix output
Usage:
  python3 convert_png_to_mnist_format.py input.png --brightness 3
  python3 convert_png_to_mnist_format.py input.png --scale_text_output
"""

import cv2
import numpy as np
from PIL import Image
import argparse
import sys
import os

def print_matrix(matrix, scale=False):
    if scale:
        # 0.00 to 1.00
        formatted = matrix.astype(float) / 255.0
        for row in formatted:
            print(" ".join(f"{x:.2f}" for x in row))
    else:
        # 0 to 255
        for row in matrix:
            print(" ".join(f"{x:3d}" for x in row))

def handwritten_to_mnist(input_path, brightness=1.0, scale_text_output=False):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Cannot read {input_path}")
        sys.exit(1)

    # Invert + Otsu threshold
    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find largest contour
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Error: No digit found")
        sys.exit(1)

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Crop with margin
    margin = max(10, int(max(w, h) * 0.3))
    digit = bin_img[max(y-margin,0):y+h+margin, max(x-margin,0):x+w+margin]

    # Resize to 20x20
    digit_20 = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

    # Center in 28x28
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[4:24, 4:24] = digit_20

    # Apply brightness multiplier
    if brightness != 1.0:
        canvas = (canvas.astype(np.float32) * brightness).clip(0, 255).astype(np.uint8)

    # Save PNG
    output_path = os.path.splitext(input_path)[0] + "_mnist_28x28.png"
    Image.fromarray(canvas).save(output_path)

    # Print success
    print(f"SUCCESS → {output_path}")
    print(f"   Brightness: ×{brightness} | Max pixel: {canvas.max()} | Non-zero: {np.count_nonzero(canvas)}")
    print("\n28×28 Pixel Matrix:")
    print_matrix(canvas, scale=scale_text_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert handwritten digit to MNIST 28x28 format")
    parser.add_argument("input", help="Input image file (PNG/JPG)")
    parser.add_argument("--brightness", type=float, default=1.0,
                        help="Brightness multiplier (e.g. 3 for 3x brighter)")
    parser.add_argument("--scale_text_output", action="store_true",
                        help="Print matrix as 0.00–1.00 instead of 0–255")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    handwritten_to_mnist(
        input_path=args.input,
        brightness=args.brightness,
        scale_text_output=args.scale_text_output
    )