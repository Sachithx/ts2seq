# QUICK REFERENCE GUIDE

## 🚀 Quick Start (Copy-Paste Commands)

### 1. Basic Training
```bash
python main.py
```

### 2. Fast Test (5 epochs)
```bash
python main.py --epochs 5 --batch-size 32
```

### 3. High Performance Training
```bash
python main.py --epochs 30 --batch-size 64 --lr 2e-3 --warmup-epochs 3
```

### 4. Fine-tune Everything
```bash
python main.py --no-freeze-encoder --epochs 15 --lr 5e-5 --batch-size 16
```

---

## 📊 Monitoring

### View Training Progress
```bash
# TensorBoard (real-time)
tensorboard --logdir checkpoints/optimized/logs

# Generate plots
python -c "
from visualize import plot_training_history
plot_training_history('checkpoints/optimized/training_results.json', 'checkpoints/optimized')
"
```

### Generate Report
```bash
python -c "
from visualize import create_training_report
create_training_report('checkpoints/optimized', 'report.md')
"
```

---

## 🔍 Profiling & Benchmarking

### Quick Profile
```bash
python -c "
from profiler import PerformanceProfiler
from models.encoder_cls import EncoderClassifier
from data_loader import create_har_dataloaders

train_loader, _, _ = create_har_dataloaders(batch_size=32)
model = EncoderClassifier(num_classes=6, pretrained_encoder_path='<path>', freeze_encoder=True)

profiler = PerformanceProfiler(model, train_loader)
profiler.run_full_profile('profile_results.json')
"
```

### Batch Size Benchmark
```bash
python -c "
from profiler import PerformanceProfiler
from models.encoder_cls import EncoderClassifier
from data_loader import create_har_dataloaders

train_loader, _, _ = create_har_dataloaders(batch_size=32)
model = EncoderClassifier(num_classes=6, pretrained_encoder_path='<path>')

profiler = PerformanceProfiler(model, train_loader)
profiler.benchmark_batch_sizes([16, 32, 64, 128])
"
```

---

## 💾 Checkpoint Management

### Load Best Model
```python
import torch
from models.encoder_cls import EncoderClassifier

model = EncoderClassifier(num_classes=6)
checkpoint = torch.load('checkpoints/optimized/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Resume Training
```python
from train_optimized import OptimizedTrainer

# Load checkpoint
checkpoint = torch.load('checkpoints/optimized/latest_checkpoint.pth')

# Create trainer (same config as before)
trainer = OptimizedTrainer(...)

# Load states
trainer.model.load_state_dict(checkpoint['model_state_dict'])
trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
trainer.scheduler.current_epoch = checkpoint['scheduler_state_dict']['current_epoch']

# Continue training
trainer.train()
```

---

## 🎯 Evaluation

### Evaluate Test Set
```python
from visualize import evaluate_model, plot_confusion_matrix, generate_classification_report

# Load model
model = ...  # Load your model
test_loader = ...  # Your test loader

# Evaluate
y_true, y_pred = evaluate_model(model, test_loader, device='cuda')

# Confusion matrix
plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png')

# Classification report
generate_classification_report(y_true, y_pred, save_path='classification_report.txt')
```

---

## ⚙️ Common Configurations

### Memory-Constrained (Low GPU Memory)
```bash
python main.py \
    --batch-size 8 \
    --grad-accum-steps 4 \
    --num-workers 2
```

### Maximum Throughput
```bash
python main.py \
    --batch-size 128 \
    --num-workers 8 \
    --grad-accum-steps 1
```

### Best Accuracy
```bash
python main.py \
    --epochs 50 \
    --batch-size 32 \
    --lr 5e-4 \
    --weight-decay 5e-4 \
    --dropout 0.2 \
    --hidden-dims 1024 512 256 \
    --early-stopping-patience 15
```

### Fastest Training
```bash
python main.py \
    --epochs 10 \
    --batch-size 64 \
    --lr 2e-3 \
    --warmup-epochs 1 \
    --early-stopping-patience 3
```

---

## 🐛 Troubleshooting

### OOM Error
```bash
# Reduce batch size
python main.py --batch-size 16

# Or use gradient accumulation
python main.py --batch-size 8 --grad-accum-steps 4
```

### Slow Training
```bash
# Increase workers
python main.py --num-workers 8

# Larger batch size
python main.py --batch-size 64
```

### Poor Convergence
```bash
# Lower learning rate + warmup
python main.py --lr 5e-4 --warmup-epochs 5

# More regularization
python main.py --weight-decay 5e-4 --dropout 0.2
```

### Overfitting
```bash
# Add regularization
python main.py --weight-decay 1e-3 --dropout 0.3

# Data augmentation (already enabled by default)
python main.py  # augmentation is on by default

# Early stopping
python main.py --early-stopping-patience 5
```

---

## 📈 Performance Expectations

### Training Speed (A100 GPU)
| Batch Size | Throughput | Time/Epoch |
|------------|------------|------------|
| 32         | 150-200/s  | 2-3 min    |
| 64         | 250-300/s  | 1-2 min    |
| 128        | 400-500/s  | <1 min     |

### Memory Usage
| Config              | Memory  |
|---------------------|---------|
| BS=32, Frozen       | 4-6 GB  |
| BS=64, Frozen       | 8-10 GB |
| BS=32, Unfrozen     | 10-12 GB|

### Expected Accuracy
- Validation: 90-95%
- Test: 88-93%
- Training time (20 epochs): 40-60 min

---

## 🔧 Useful Python Snippets

### Check GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Count Parameters
```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,}")
print(f"Trainable: {trainable:,}")
```

### Get Predictions
```python
model.eval()
with torch.no_grad():
    outputs = model(images)
    _, predictions = outputs.max(1)
```

---

## 📝 File Structure Reference

```
project/
├── main.py                  # Main training script
├── train_optimized.py       # Optimized trainer
├── data_loader.py          # Data loading
├── profiler.py             # Performance profiling
├── visualize.py            # Visualization tools
├── examples.py             # Example scenarios
├── models/
│   └── encoder_cls.py      # Model architecture
└── checkpoints/
    └── optimized/
        ├── best_model.pth
        ├── latest_checkpoint.pth
        ├── training_results.json
        └── logs/
```

---

## 🎓 Best Practices

1. **Always start with a quick test run** (5 epochs) to verify setup
2. **Monitor TensorBoard** during training
3. **Save checkpoints frequently** (automatic)
4. **Use early stopping** to prevent overfitting
5. **Profile before scaling up** batch size
6. **Freeze encoder first**, fine-tune later if needed
7. **Use gradient accumulation** for effective larger batches
8. **Enable AMP** for faster training (default)

---

**Need help?** Check README.md for detailed documentation!
