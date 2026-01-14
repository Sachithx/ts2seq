# OPTIMIZED HAR CLASSIFICATION TRAINING

## Quick Start Guide

This is an optimized end-to-end training pipeline for HAR (Human Activity Recognition) classification using a pretrained Pix2Seq ViT encoder.

### Features

✨ **Performance Optimizations**
- Mixed Precision Training (AMP) - ~2x speedup
- Gradient Accumulation - Train with larger effective batch sizes
- Gradient Clipping - Stable training
- cuDNN Benchmark Mode - Auto-tuned convolutions
- TF32 Precision - Faster matmul on Ampere+ GPUs
- Torch.compile - JIT compilation (PyTorch 2.0+)

📊 **Training Features**
- Learning Rate Warmup + Cosine Annealing
- Early Stopping with patience
- Label Smoothing (0.1)
- Data Augmentation
- Automatic checkpoint management
- TensorBoard logging
- Comprehensive metrics tracking

🔍 **Monitoring**
- Real-time training progress with tqdm
- Per-class accuracy
- Throughput metrics (samples/sec)
- Memory profiling
- Performance benchmarking

---

## Installation

```bash
# Required packages
pip install torch torchvision tqdm numpy

# Optional: TensorBoard for visualization
pip install tensorboard
```

---

## Usage

### 1. Basic Training

```bash
# Train with default settings
python main.py
```

### 2. Custom Configuration

```bash
# Train with custom hyperparameters
python main.py \
    --mode flatten \
    --epochs 30 \
    --batch-size 64 \
    --lr 5e-4 \
    --warmup-epochs 3 \
    --hidden-dims 512 256 \
    --save-dir checkpoints/experiment_1
```

### 3. Advanced Options

```bash
# Full training with gradient accumulation and custom settings
python main.py \
    --mode flatten \
    --epochs 50 \
    --batch-size 32 \
    --grad-accum-steps 2 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --warmup-epochs 5 \
    --max-grad-norm 1.0 \
    --early-stopping-patience 10 \
    --dropout 0.2 \
    --save-dir checkpoints/experiment_2
```

### 4. Fine-tune Encoder

```bash
# Unfreeze encoder for fine-tuning
python main.py \
    --no-freeze-encoder \
    --lr 5e-5 \
    --epochs 20 \
    --batch-size 16
```

---

## Command Line Arguments

### Data Arguments
- `--mode`: Channel processing mode (`flatten`, `average`, `first`)
- `--data-dir`: Path to data directory
- `--encoder-path`: Path to pretrained encoder weights
- `--num-workers`: Number of data loader workers (default: 4)
- `--no-augmentation`: Disable data augmentation

### Training Arguments
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Training batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--weight-decay`: Weight decay (default: 1e-4)
- `--warmup-epochs`: Number of warmup epochs (default: 2)

### Model Arguments
- `--freeze-encoder`: Freeze encoder weights (default: True)
- `--no-freeze-encoder`: Unfreeze encoder for fine-tuning
- `--hidden-dims`: Hidden dimensions for classifier (default: [512])
- `--dropout`: Dropout rate (default: 0.1)

### Optimization Arguments
- `--grad-accum-steps`: Gradient accumulation steps (default: 1)
- `--max-grad-norm`: Max gradient norm for clipping (default: 1.0)
- `--no-amp`: Disable mixed precision training

### Other Arguments
- `--early-stopping-patience`: Early stopping patience (default: 7)
- `--save-dir`: Checkpoint directory (default: checkpoints/optimized)
- `--log-interval`: Log interval (default: 50)
- `--device`: Device to use (`cuda` or `cpu`)

---

## Performance Profiling

```python
# Profile training performance
from profiler import PerformanceProfiler
from models.encoder_cls import EncoderClassifier
from data_loader import create_har_dataloaders

# Create model and data
model = EncoderClassifier(num_classes=6)
train_loader, _, _ = create_har_dataloaders(batch_size=32)

# Run profiling
profiler = PerformanceProfiler(model, train_loader, device='cuda')
results = profiler.run_full_profile(save_path='profile_results.json')

# Benchmark different batch sizes
batch_results = profiler.benchmark_batch_sizes([8, 16, 32, 64, 128])

# Compare AMP performance
from profiler import compare_amp_performance
amp_results = compare_amp_performance(model, train_loader)
```

---

## Expected Performance

### Training Speed (with AMP on A100)
- Batch size 32: ~150-200 samples/sec
- Batch size 64: ~250-300 samples/sec
- Full epoch: ~2-3 minutes (flatten mode)

### Memory Usage
- Encoder frozen, batch size 32: ~4-6 GB
- Encoder frozen, batch size 64: ~8-10 GB
- Encoder unfrozen, batch size 32: ~10-12 GB

### Accuracy (Expected)
- Validation accuracy: 90-95%
- Test accuracy: 88-93%
- Training time (20 epochs): 40-60 minutes

---

## Monitoring with TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir checkpoints/optimized/logs

# Open browser to http://localhost:6006
```

---

## Checkpoint Structure

```
checkpoints/optimized/
├── best_model.pth          # Best model by validation accuracy
├── latest_checkpoint.pth   # Latest epoch checkpoint
├── training_results.json   # Training history and results
└── logs/                   # TensorBoard logs
    └── events.out.tfevents...
```

### Loading a Checkpoint

```python
import torch
from models.encoder_cls import EncoderClassifier

# Load model
model = EncoderClassifier(num_classes=6)
checkpoint = torch.load('checkpoints/optimized/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Access training info
print(f"Epoch: {checkpoint['epoch']}")
print(f"Val Acc: {checkpoint['val_acc']:.2f}%")
print(f"History: {checkpoint['history']}")
```

---

## Optimization Tips

### 1. Maximize Throughput
```bash
# Use largest batch size that fits in memory
python main.py --batch-size 64

# Or use gradient accumulation for effective larger batch
python main.py --batch-size 32 --grad-accum-steps 4  # Effective: 128
```

### 2. Faster Convergence
```bash
# Longer warmup + higher learning rate
python main.py --lr 2e-3 --warmup-epochs 5

# Add more capacity to classifier
python main.py --hidden-dims 1024 512 256
```

### 3. Better Generalization
```bash
# More regularization
python main.py --weight-decay 5e-4 --dropout 0.2

# Longer training with early stopping
python main.py --epochs 100 --early-stopping-patience 15
```

### 4. Fine-tuning Strategy
```bash
# Stage 1: Train classifier only (faster)
python main.py --freeze-encoder --epochs 20 --lr 1e-3

# Stage 2: Fine-tune entire model (better accuracy)
python main.py --no-freeze-encoder --epochs 10 --lr 5e-5
```

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python main.py --batch-size 16

# Use gradient accumulation
python main.py --batch-size 16 --grad-accum-steps 2

# Disable AMP (uses slightly more memory)
python main.py --no-amp
```

### Slow Training
```bash
# Increase num workers
python main.py --num-workers 8

# Enable AMP if not already
python main.py  # AMP is enabled by default

# Reduce logging
python main.py --log-interval 100
```

### Not Converging
```bash
# Lower learning rate
python main.py --lr 5e-4

# Add warmup
python main.py --warmup-epochs 5

# Check gradient norm
python main.py --max-grad-norm 0.5
```

---

## File Structure

```
.
├── main.py                 # Main training script
├── train_optimized.py      # Optimized trainer class
├── profiler.py            # Performance profiling utilities
├── data_loader.py         # Data loading and augmentation
├── models/
│   └── encoder_cls.py     # Model architecture
├── checkpoints/           # Saved models
└── README.md             # This file
```

---

