#!/usr/bin/env python3
"""
MAIN TRAINING SCRIPT - OPTIMIZED HAR CLASSIFICATION
Usage: python main.py [--mode flatten] [--epochs 20] [--batch-size 32]
"""

import argparse
import torch
from pathlib import Path

from train_optimized import OptimizedTrainer, setup_training_environment
from data_loader import create_har_dataloaders
from models.encoder_cls import EncoderClassifier


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train HAR classifier with Pix2Seq encoder')
    
    # Data arguments
    parser.add_argument('--mode', type=str, default='flatten',
                       choices=['flatten', 'average', 'first'],
                       help='Channel processing mode')
    parser.add_argument('--data-dir', type=str,
                       default='/home/AD/sachith/ts2seq/data/multichannel_images',
                       help='Path to data directory')
    parser.add_argument('--encoder-path', type=str,
                       default='/home/AD/sachith/ts2seq/data/HAR_pretrained/google_vit_encoder/encoder_weights.pth',
                       help='Path to pretrained encoder weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                       help='Number of warmup epochs')
    
    # Model arguments
    parser.add_argument('--freeze-encoder', action='store_true', default=True,
                       help='Freeze encoder weights')
    parser.add_argument('--no-freeze-encoder', action='store_false', dest='freeze_encoder',
                       help='Do not freeze encoder weights')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[64],
                       help='Hidden dimensions for classification head')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--patch-size', type=int, default=16,
                       help='Patch size for the encoder')
                           
    # Optimization arguments
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                       help='Max gradient norm for clipping')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision training')
    
    # Data loader arguments
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--data-pct', type=int, default=5,
                       help='Percentage of training data to use (1-100)')
    
    # Early stopping
    parser.add_argument('--early-stopping-patience', type=int, default=7,
                       help='Early stopping patience')
    
    # Logging and checkpointing
    parser.add_argument('--save-dir', type=str, default='checkpoints/optimized',
                       help='Directory to save checkpoints')
    parser.add_argument('--log-interval', type=int, default=50,
                       help='Log interval for training progress')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup environment
    print("\n" + "="*80)
    print("OPTIMIZED HAR CLASSIFIER TRAINING")
    print("="*80)
    setup_training_environment()
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("\n⚠ CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Create dataloaders
    print("\n[1/4] Creating dataloaders...")
    train_loader, val_loader, test_loader = create_har_dataloaders(
        data_dir=args.data_dir,
        mode=args.mode, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=not args.no_augmentation,
        train_percentage=args.data_pct,
        force_resample=True,
        subset_indices_path=f'subset_indices/har_{args.data_pct}pct_seed42.pt'  # Will save here
    )
    
    # Create model
    print("\n[2/4] Creating model...")
    model = EncoderClassifier(
        patch_size=args.patch_size,
        num_classes=6,
        pretrained_encoder_path=args.encoder_path,
        freeze_encoder=args.freeze_encoder,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        image_size=224
    )
    
    # Compile model (PyTorch 2.0+)
    if hasattr(torch, 'compile') and device == 'cuda':
        print("\n✓ Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
    
    # Create trainer
    print("\n[3/4] Initializing trainer...")
    trainer = OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        early_stopping_patience=args.early_stopping_patience,
        save_dir=args.save_dir,
        use_amp=not args.no_amp and device == 'cuda',
        log_interval=args.log_interval
    )
    
    # Train
    print("\n[4/4] Starting training...")
    results = trainer.train()
    
    # Print final summary
    print("\n" + "="*80)
    print("TRAINING SESSION SUMMARY")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs trained: {results['best_epoch'] + 1}")
    print(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
    print(f"Test accuracy: {results['test_acc']:.2f}%")
    print(f"Total time: {results['total_time_minutes']:.1f} minutes")
    print(f"Checkpoints saved to: {args.save_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
