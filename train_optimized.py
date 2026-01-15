"""
OPTIMIZED END-TO-END TRAINING PIPELINE FOR HAR CLASSIFICATION
Features:
- Mixed precision training (AMP)
- Gradient accumulation
- Gradient clipping
- Learning rate warmup
- Early stopping
- Checkpoint management
- TensorBoard logging
- Performance profiling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import json
import time
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from models.encoder_cls import EncoderClassifier

# Optional: TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠ TensorBoard not available. Install with: pip install tensorboard")


class MetricsTracker:
    """Track and compute training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = defaultdict(list)
        self.batch_times = []
        self.start_time = None
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(value)
    
    def get_average(self, key):
        if key in self.metrics and len(self.metrics[key]) > 0:
            return np.mean(self.metrics[key])
        return 0.0
    
    def start_batch(self):
        self.start_time = time.time()
    
    def end_batch(self):
        if self.start_time:
            self.batch_times.append(time.time() - self.start_time)
    
    def get_throughput(self, batch_size):
        if len(self.batch_times) > 0:
            avg_time = np.mean(self.batch_times)
            return batch_size / avg_time if avg_time > 0 else 0
        return 0


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
        
        return self.should_stop


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class OptimizedTrainer:
    """Optimized trainer with all performance enhancements."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device='cuda',
        num_epochs=20,
        learning_rate=1e-3,
        weight_decay=1e-4,
        warmup_epochs=2,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        early_stopping_patience=7,
        save_dir='checkpoints/optimized',
        use_amp=True,
        log_interval=50
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and device == 'cuda'
        self.log_interval = log_interval
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Criterion and optimizer
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=num_epochs
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=0.001,
            mode='max'
        )
        
        # TensorBoard
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.save_dir / 'logs'))
        
        # Metrics
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        self._print_config()
    
    def _print_config(self):
        """Print training configuration."""
        print("\n" + "="*80)
        print("OPTIMIZED TRAINING CONFIGURATION")
        print("="*80)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\nModel:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {total_params - trainable_params:,}")
        
        print(f"\nData:")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Test batches: {len(self.test_loader)}")
        print(f"  Batch size: {self.train_loader.batch_size}")
        
        print(f"\nTraining:")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Weight decay: {self.optimizer.defaults['weight_decay']}")
        print(f"  Warmup epochs: {self.scheduler.warmup_epochs}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  Max grad norm: {self.max_grad_norm}")
        print(f"  Label smoothing: 0.1")
        
        print(f"\nOptimizations:")
        print(f"  Mixed precision (AMP): {self.use_amp}")
        print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  TF32: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  Early stopping patience: {self.early_stopping.patience}")
        
        print(f"\nDevice: {self.device}")
        if self.device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        print("="*80 + "\n")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f'Epoch {epoch+1}/{self.num_epochs} [TRAIN]',
            ncols=120
        )
        
        for batch_idx, (images, labels) in pbar:
            self.train_metrics.start_batch()
            
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights (with gradient accumulation)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum()
            accuracy = 100. * correct / labels.size(0)
            
            # Update metrics
            self.train_metrics.update(
                loss=loss.item() * self.gradient_accumulation_steps,
                accuracy=accuracy
            )
            self.train_metrics.end_batch()
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                avg_loss = self.train_metrics.get_average('loss')
                avg_acc = self.train_metrics.get_average('accuracy')
                throughput = self.train_metrics.get_throughput(images.size(0))
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{avg_acc:.2f}%',
                    'samples/s': f'{throughput:.1f}'
                })
        
        # Epoch statistics
        avg_loss = self.train_metrics.get_average('loss')
        avg_acc = self.train_metrics.get_average('accuracy')
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        self.val_metrics.reset()
        
        pbar = tqdm(
            self.val_loader,
            desc=f'Epoch {epoch+1}/{self.num_epochs} [VAL]  ',
            ncols=120
        )
        
        for images, labels in pbar:
            self.val_metrics.start_batch()
            
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum()
            accuracy = 100. * correct / labels.size(0)
            
            # Update metrics
            self.val_metrics.update(loss=loss, accuracy=accuracy)
            self.val_metrics.end_batch()
            
            # Update progress bar
            avg_loss = self.val_metrics.get_average('loss')
            avg_acc = self.val_metrics.get_average('accuracy')
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{avg_acc:.2f}%'
            })
        
        avg_loss = self.val_metrics.get_average('loss')
        avg_acc = self.val_metrics.get_average('accuracy')
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def test(self):
        """Test the model."""
        self.model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        
        # Per-class accuracy
        class_correct = [0] * 6
        class_total = [0] * 6
        
        pbar = tqdm(self.test_loader, desc='Testing', ncols=120)
        
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            test_loss += loss.item()
            
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1
            
            pbar.set_postfix({
                'acc': f'{100.*test_correct/test_total:.2f}%'
            })
        
        test_acc = 100. * test_correct / test_total
        test_loss /= len(self.test_loader)
        
        # Print per-class accuracy
        print("\nPer-class accuracy:")
        class_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 
                      'Sitting', 'Standing', 'Laying']
        for i in range(6):
            if class_total[i] > 0:
                acc = 100. * class_correct[i] / class_total[i]
                print(f"  {class_names[i]:20s}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        
        return test_loss, test_acc
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': int(epoch),  # ← Convert to Python int
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': {
                'current_epoch': int(self.scheduler.current_epoch),  # ← Convert to int
                'base_lr': float(self.scheduler.base_lr)  # ← Convert to float
            },
            'val_acc': float(val_acc),  # ← Convert to Python float
            'best_val_acc': float(self.best_val_acc),  # ← Add this
            'history': {
                'train_loss': [float(x) for x in self.history['train_loss']],  # ← Convert list elements
                'train_acc': [float(x) for x in self.history['train_acc']],
                'val_loss': [float(x) for x in self.history['val_loss']],
                'val_acc': [float(x) for x in self.history['val_acc']],
                'learning_rates': [float(x) for x in self.history['learning_rates']],
                'epoch_times': [float(x) for x in self.history['epoch_times']]
            },
            'config': {
                'learning_rate': float(self.learning_rate),
                'num_epochs': int(self.num_epochs),
                'gradient_accumulation_steps': int(self.gradient_accumulation_steps),
                'use_amp': bool(self.use_amp)
            }
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80 + "\n")
        
        total_start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update scheduler
            current_lr = self.scheduler.step()
            
            # Epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            self.history['epoch_times'].append(epoch_time)
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            # Print summary
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{self.num_epochs} Summary:")
            print(f"{'='*80}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Epoch Time: {epoch_time:.1f}s")
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            self.save_checkpoint(epoch, val_acc, is_best)
            
            # Early stopping
            if self.early_stopping(val_acc):
                print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                print(f"  Best val acc: {self.best_val_acc:.2f}% at epoch {self.best_epoch+1}")
                break
            
            print(f"{'='*80}\n")
        
        # # Training complete
        total_time = time.time() - total_start_time
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (epoch {self.best_epoch+1})")
        
        # Load best model and test
        print("\nLoading best model for testing...")


        model = EncoderClassifier(
            patch_size=8,
            num_classes=6,
            pretrained_encoder_path="/home/AD/sachith/ts2seq/data/HAR_pretrained/own_model/encoder_weights.pth",
            freeze_encoder=True,
            hidden_dims=[512, 256],
            dropout=0.3,
            image_size=224
        )
        if hasattr(torch, 'compile'):
            print("\n✓ Compiling model with torch.compile...")
            model = torch.compile(model, mode='reduce-overhead')
        checkpoint = torch.load("/home/AD/sachith/ts2seq/checkpoints/optimized/best_model.pth", weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_acc = self.test()
        
        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}")
        print(f"Best Val Accuracy:  {self.best_val_acc:.2f}%")
        print(f"Test Accuracy:      {test_acc:.2f}%")
        print(f"Test Loss:          {test_loss:.4f}")
        print(f"{'='*80}\n")
        
        # Save results
        results = {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'total_time_minutes': total_time / 60,
            'history': self.history,
            'config': {
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'use_amp': self.use_amp,
                'warmup_epochs': self.scheduler.warmup_epochs
            }
        }
        
        with open(self.save_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        if self.writer:
            self.writer.close()
        
        return results


def setup_training_environment():
    """Setup optimal training environment."""
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set deterministic for reproducibility (optional)
    # torch.backends.cudnn.deterministic = True
    # torch.manual_seed(42)
    # np.random.seed(42)
    
    print("✓ Training environment configured")
    print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"  TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")


if __name__ == '__main__':
    # This file is meant to be imported
    # See main.py for usage
    pass
