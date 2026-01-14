# %%
# ============================================================================
# TRAINING SCRIPT
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json

from data_loader import create_har_dataloaders
from models.encoder_cls import EncoderClassifier

# Enable these at the top of your script
torch.backends.cudnn.benchmark = True  # Auto-tune conv operations
torch.backends.cuda.matmul.allow_tf32 = True  # Faster matmul on Ampere+ GPUs
torch.backends.cudnn.allow_tf32 = True


# Use mixed precision training
from torch.amp import autocast
from torch.cuda.amp import GradScaler

def train_har_classifier(
    mode='flatten',
    batch_size=32,
    num_epochs=20,
    learning_rate=1e-3,
    device='cuda',
    save_dir='checkpoints/har_classifier',
    pretrained_encoder_path='/home/sachithxcviii/ts2seq/data/HAR/extracted_encoder/encoder_weights.pth',
    data_dir='/home/sachithxcviii/ts2seq/data/HAR/multichannel_images'
):
    """
    Train HAR classifier with pretrained Pix2Seq encoder.
    
    Args:
        mode: 'flatten', 'average', or 'first'
        batch_size: Training batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: 'cuda' or 'cpu'
        save_dir: Directory to save checkpoints
        pretrained_encoder_path: Path to pretrained encoder weights
        data_dir: Directory containing HAR data
    """
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("TRAINING HAR CLASSIFIER WITH PIX2SEQ ENCODER")
    print("="*80)
    print(f"Configuration:")
    print(f"  Mode: {mode}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    
    # Create dataloaders
    print("\n[1/5] Creating dataloaders...")
    train_loader, val_loader, test_loader = create_har_dataloaders(
        mode=mode,
        batch_size=batch_size,
        use_augmentation=True,
        data_dir=data_dir
    )
    
    # Create model
    print("\n[2/5] Loading model with pretrained encoder...")
    model = EncoderClassifier(
        num_classes=6,
        pretrained_encoder_path=pretrained_encoder_path,
        freeze_encoder=True,
        hidden_dims=[512],
        dropout=0.1,
        image_size=224
    )
    model = model.to(device)
    # Compile model (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode='reduce-overhead')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Mixed precision scaler
    scaler = GradScaler()

    # Training loop
    print("\n[3/5] Training...")
    best_val_acc = 0.0
    train_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward
            optimizer.zero_grad()
            # Mixed precision forward pass
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Stats
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_accs.append(val_acc)
        
        # Update scheduler
        scheduler.step()
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Acc: {val_acc:.2f}%')
        print(f'  LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'mode': mode
            }, f'{save_dir}/best_model.pth')
            print(f'  ✓ Saved best model (val_acc: {val_acc:.2f}%)')
    
    # Test evaluation
    print("\n[4/5] Evaluating on test set...")
    checkpoint = torch.load(f'{save_dir}/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * test_correct / test_total
    
    # Save training history
    print("\n[5/5] Saving results...")
    results = {
        'mode': mode,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'train_losses': train_losses,
        'val_accs': val_accs,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    
    with open(f'{save_dir}/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Model saved to: {save_dir}/best_model.pth")
    
    return model, results
