# MNIST Digit Testing: Exploring AI Hallucinations

A research project investigating whether training neural networks to recognize non-digit inputs (whitespace, letters) can reduce hallucinations and improve digit classification accuracy on MNIST data.

## Project Background

This project explores a fundamental question about AI hallucinations: **Can we reduce digit-to-digit misclassification errors by training neural networks to also recognize what digits are NOT?**

When a neural network is trained only on digits (0-9), it will always predict a digit, even when given a letter "Q" or blank space. This creates hallucinations - the model must output something from its trained classes. By training on additional classes (whitespace and non-numbers), we test whether the model develops better internal representations that improve digit classification accuracy.

**Spoiler**: The results show that training on additional classes does NOT improve digit classification accuracy. On average, accuracy decreases by 0.2-0.3% when training includes non-digit classes, suggesting that computational resources are better spent on the target task.

Read the full analysis: [AI Hallucinations and MNIST Digits](https://blog.agafamily.com/2025/11/07/ai-hallucinations-and-mnist-digits/)

## Architecture

The project uses a 3-layer convolutional neural network:
- **Input layer**: 784 neurons (28×28 pixel images)
- **Hidden layer 1**: 256 neurons
- **Hidden layer 2**: 128 neurons  
- **Output layer**: 10-12 neurons (depending on classes trained)
- **Total**: ~1,200 neurons with ~235,000 synapses

## Key Features

- **Flexible class training**: Train on any combination of digits (d), whitespace (w), and non-numbers/letters (nan)
- **Automated testing**: Run multiple configurations across different epoch counts
- **Detailed metrics**: Track Type I errors (false positives) and Type II errors (false negatives/hallucinations)
- **Visualization**: View misclassified samples with softmax probability distributions
- **Excel reporting**: Generate comprehensive accuracy reports with betterment analysis

## Installation

### Prerequisites

- Python 3.7+
- PyTorch with MPS/CUDA support (for Mac or NVIDIA GPU acceleration)
- Required packages:

```bash
pip install torch torchvision numpy pandas openpyxl
```

### Dataset Download

The scripts automatically download MNIST and EMNIST datasets on first run.

## Usage

### Basic Training

Train on digits only:
```bash
./mnist_digits_10.py --train_classes d --test_classes d --epochs 10
```

Train on digits, whitespace, and non-numbers:
```bash
./mnist_digits_10.py --train_classes d,w,nan --test_classes d --epochs 10
```

### Visualization and Analysis

Show softmax probabilities for misclassified samples:
```bash
./mnist_digits_10.py --train_classes d --test_classes d --epochs 10 \
  --show_softmax_output_probabilities --visualize 5 --test_seed 19
```

### Automated Testing

Run comprehensive tests across multiple configurations:

```bash
# Test all configurations for 50 epochs
./automate_mnist.py --epochs 50

# Custom batch size and sample counts
./automate_mnist.py --epochs 100 --batch_size 64 --train_samples 30000
```

The automation script tests four configurations:
1. **Config 1**: Train on digits, whitespace, and letters (d,w,nan)
2. **Config 2**: Train on digits and whitespace (d,w)
3. **Config 3**: Train on digits and letters (d,nan)
4. **Config 4**: Train on digits only (d)

Results are saved to an Excel file with accuracy metrics and betterment analysis comparing each configuration to the baseline (digits only).

## Command-Line Arguments

### mnist_digits_10.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_samples` | 50000 | Number of training samples |
| `--test_samples` | 10000 | Number of test samples |
| `--epochs` | 10 | Training epochs |
| `--batch_size` | 128 | Batch size |
| `--train_classes` | d | Classes for training: d,w,nan |
| `--test_classes` | d | Classes for testing: d,w,nan |
| `--train_seed` | 42 | Random seed for training data |
| `--test_seed` | 42 | Random seed for test data |
| `--visualize` | - | Show N misclassified samples |
| `--show_softmax_output_probabilities` | False | Show detailed softmax output |

### automate_mnist.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Number of epochs to train |
| `--batch_size` | 128 | Batch size for training |
| `--train_samples` | 50000 | Number of training samples |
| `--test_samples` | 10000 | Number of test samples |
| `--train_seed` | 1 | Random seed for training |
| `--test_seed` | 1 | Random seed for testing |

## Understanding the Results

### Error Types

- **Type I errors**: False positives (predicting a digit is not a digit)
- **Type II errors**: False negatives/hallucinations (misclassifying one digit as another)

Example output:
```
Overall accuracy on 1000 test images: 98.00%
Total Type I errors: 2 / 1000 (0.20%)
Total Type II errors: 20 / 1000 (2.00%)

Detailed breakdown:
Class 0: 97.9% (94/96) correct
Class 1: 100.0% (102/102) correct
Class 8: 93.3% (84/90) correct
```

### Key Findings

Based on extensive testing across multiple configurations:

1. **Training on additional classes REDUCES accuracy** for digit classification:
   - Whitespace: -0.3% average accuracy
   - Letters (NaN): -0.15% average accuracy  
   - Both: -0.2% average accuracy

2. **Computational resources matter**: Better to spend compute power on more epochs/samples of the target task rather than training on expanded classes

3. **Hallucinations persist**: Even with expanded training classes, Type II errors (digit-to-digit misclassifications) occur at similar rates

## File Structure

```
.
├── mnist_digits_10.py      # Main training script
├── automate_mnist.py        # Automation script for batch testing
├── README.md                # This file
└── results/                 # Generated Excel files (created at runtime)
```

## Output Files

The automation script generates Excel files with:
- Accuracy results for each configuration across all epochs
- Betterment tables comparing configurations
- Summary statistics (best accuracy, final accuracy, averages)
- Timestamp-based filenames for tracking multiple runs

Example filename: `mnist_accuracy_results_100epochs_20250107_143052.xlsx`

## Hardware Requirements

- **CPU**: Works on any modern processor
- **GPU**: Recommended for faster training
  - NVIDIA GPU with CUDA support
  - Apple Silicon with MPS (Metal Performance Shaders)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: ~200MB for datasets

## Performance Notes

Training times vary by hardware:
- **Apple M1/M2 (MPS)**: ~0.03ms per sample
- **CPU only**: ~0.5-1ms per sample
- **CUDA GPU**: ~0.01-0.02ms per sample

For 100 epochs with 50,000 samples: expect 2-6 hours on CPU, 30-60 minutes on GPU.

## Contributing

This is a research/educational project. Feel free to:
- Experiment with different architectures
- Test other datasets
- Explore different class combinations
- Share your findings

## License

MIT License - feel free to use this code for educational and research purposes.

## Citation

If you use this code in your research or find the methodology interesting, please reference:

```
Neaga, Neville (2025). "AI Hallucinations and MNIST Digits: Exploring Whether 
Training for Non-Digits Improves Digit Classification." 
https://blog.agafamily.com/2025/11/07/ai-hallucinations-and-mnist-digits/
```

## Acknowledgments

- MNIST dataset: Yann LeCun et al.
- EMNIST dataset: Gregory Cohen et al.
- PyTorch: Facebook AI Research

## Author

**Neville Neaga**  
Blog: [blog.agafamily.com](https://blog.agafamily.com)  
GitHub: [@CiscoNeville](https://github.com/CiscoNeville)

---

*"The curious paradox: teaching a neural network what something is NOT doesn't help it better identify what something IS."*
