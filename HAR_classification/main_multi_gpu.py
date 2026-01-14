#!/usr/bin/env python3
"""
MAIN MULTI-GPU TRAINING SCRIPT
Supports both DistributedDataParallel (DDP) and DataParallel (DP)
"""

import argparse
import torch
import os

from train_optimized import setup_training_environment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multi-GPU HAR Classifier Training')
    
    # Multi-GPU arguments
    parser.add_argument('--multi-gpu', type=str, default=None,
                       choices=['ddp', 'dp', None],
                       help='Multi-GPU mode: ddp (DistributedDataParallel), dp (DataParallel), or None (single GPU)')
    parser.add_argument('--gpu-ids', type=str, default=None,
                       help='GPU IDs to use (comma-separated, e.g., "0,1,2,3"). If not set, uses all available GPUs')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for DDP (set automatically by torchrun/torch.distributed.launch)')
    
    # Data arguments
    parser.add_argument('--mode', type=str, default='flatten',
                       choices=['flatten', 'average', 'first'],
                       help='Channel processing mode')
    parser.add_argument('--data-dir', type=str,
                       default='/home/sachithxcviii/ts2seq/data/HAR/multichannel_images',
                       help='Path to data directory')
    parser.add_argument('--encoder-path', type=str,
                       default='/home/sachithxcviii/ts2seq/data/HAR/extracted_encoder/encoder_weights.pth',
                       help='Path to pretrained encoder weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size PER GPU')
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
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[512],
                       help='Hidden dimensions for classification head')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Optimization arguments
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                       help='Max gradient norm for clipping')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision training')
    
    # Data loader arguments
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers PER GPU')
    
    # Early stopping
    parser.add_argument('--early-stopping-patience', type=int, default=7,
                       help='Early stopping patience')
    
    # Logging and checkpointing
    parser.add_argument('--save-dir', type=str, default='checkpoints/multi_gpu',
                       help='Directory to save checkpoints')
    parser.add_argument('--log-interval', type=int, default=50,
                       help='Log interval for training progress')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup environment
    print("\n" + "="*80)
    print("MULTI-GPU HAR CLASSIFIER TRAINING")
    print("="*80)
    setup_training_environment()
    
    # Determine GPU configuration
    num_gpus = torch.cuda.device_count()
    print(f"\n📊 GPU Configuration:")
    print(f"  Available GPUs: {num_gpus}")
    
    if num_gpus == 0:
        print("  ❌ No GPUs available! Exiting...")
        return
    
    # Print GPU info
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Determine training mode
    if args.multi_gpu == 'ddp':
        # DistributedDataParallel
        print(f"\n✓ Using DistributedDataParallel (DDP)")
        
        # Check if launched with torchrun/torch.distributed.launch
        if args.local_rank == -1 and 'RANK' not in os.environ:
            print("\n⚠ DDP requires launching with torchrun or torch.distributed.launch")
            print("\nUsage:")
            print(f"  # Use all {num_gpus} GPUs:")
            print(f"  torchrun --nproc_per_node={num_gpus} main_multi_gpu.py --multi-gpu ddp [other args]")
            print(f"\n  # Or with torch.distributed.launch:")
            print(f"  python -m torch.distributed.launch --nproc_per_node={num_gpus} main_multi_gpu.py --multi-gpu ddp [other args]")
            print("\nAlternatively, use DataParallel (simpler but slower):")
            print(f"  python main_multi_gpu.py --multi-gpu dp [other args]")
            return
        
        # Get world size
        world_size = int(os.environ.get('WORLD_SIZE', num_gpus))
        args.world_size = world_size
        
        print(f"  World size: {world_size}")
        print(f"  Effective batch size: {args.batch_size * world_size}")
        
        # Import and run DDP training
        from train_multi_gpu import train_ddp
        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        train_ddp(local_rank, args)
    
    elif args.multi_gpu == 'dp':
        # DataParallel
        print(f"\n✓ Using DataParallel (DP)")
        
        # Determine GPU IDs
        if args.gpu_ids:
            gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
            print(f"  Using GPUs: {gpu_ids}")
        else:
            gpu_ids = list(range(num_gpus))
            print(f"  Using all {num_gpus} GPUs: {gpu_ids}")
        
        args.gpu_ids = ','.join(map(str, gpu_ids))
        print(f"  Effective batch size: {args.batch_size * len(gpu_ids)}")
        
        # Import and run DP training
        from train_multi_gpu import train_dataparallel
        train_dataparallel(args)
    
    else:
        # Single GPU
        print(f"\n✓ Using single GPU mode")
        print(f"  Batch size: {args.batch_size}")
        print("\n💡 Tip: Use --multi-gpu dp or --multi-gpu ddp for multi-GPU training")
        
        # Import and run single GPU training
        from data_loader import create_har_dataloaders
        from models.encoder_cls import EncoderClassifier
        from train_optimized import OptimizedTrainer
        
        # Create dataloaders
        print("\n[1/3] Creating dataloaders...")
        train_loader, val_loader, test_loader = create_har_dataloaders(
            data_dir=args.data_dir,
            mode=args.mode,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_augmentation=True
        )
        
        # Create model
        print("\n[2/3] Creating model...")
        model = EncoderClassifier(
            num_classes=6,
            pretrained_encoder_path=args.encoder_path,
            freeze_encoder=args.freeze_encoder,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            image_size=224
        )
        
        # Compile model (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            print("✓ Compiling model with torch.compile...")
            model = torch.compile(model, mode='reduce-overhead')
        
        # Create trainer
        print("\n[3/3] Training...")
        trainer = OptimizedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device='cuda',
            num_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            gradient_accumulation_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            early_stopping_patience=args.early_stopping_patience,
            save_dir=args.save_dir,
            use_amp=not args.no_amp,
            log_interval=args.log_interval
        )
        
        trainer.train()


if __name__ == '__main__':
    main()
