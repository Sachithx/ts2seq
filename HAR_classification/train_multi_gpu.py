"""
MULTI-GPU TRAINING SUPPORT
Supports both DataParallel (simple) and DistributedDataParallel (optimal)
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.utils.data.distributed import DistributedSampler
import os
from pathlib import Path

from train_optimized import OptimizedTrainer


class MultiGPUTrainer(OptimizedTrainer):
    """Enhanced trainer with multi-GPU support."""
    
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
        log_interval=50,
        # Multi-GPU specific
        multi_gpu_mode='ddp',  # 'ddp', 'dp', or None
        local_rank=-1,
        world_size=1,
        gpu_ids=None
    ):
        """
        Args:
            multi_gpu_mode: 'ddp' (DistributedDataParallel - recommended),
                          'dp' (DataParallel - simpler but slower),
                          None (single GPU)
            local_rank: Local rank for DDP (set by torch.distributed.launch)
            world_size: Total number of GPUs
            gpu_ids: List of GPU IDs to use (e.g., [0, 1, 2, 3])
        """
        self.multi_gpu_mode = multi_gpu_mode
        self.local_rank = local_rank
        self.world_size = world_size
        self.gpu_ids = gpu_ids
        self.is_main_process = (local_rank == -1 or local_rank == 0)
        
        # Setup distributed training if using DDP
        if multi_gpu_mode == 'ddp' and local_rank != -1:
            device = f'cuda:{local_rank}'
            self.setup_ddp(local_rank, world_size)
        
        # Initialize base trainer
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            early_stopping_patience=early_stopping_patience,
            save_dir=save_dir,
            use_amp=use_amp,
            log_interval=log_interval
        )
        
        # Wrap model for multi-GPU
        self.model = self.setup_multi_gpu_model(self.model)
        
        # Update print configuration
        if self.is_main_process:
            self._print_multi_gpu_config()
    
    def setup_ddp(self, local_rank, world_size):
        """Setup DistributedDataParallel."""
        if not dist.is_initialized():
            # Get environment variables set by torch.distributed.launch
            rank = int(os.environ.get('RANK', local_rank))
            world_size = int(os.environ.get('WORLD_SIZE', world_size))
            
            # Initialize process group
            dist.init_process_group(
                backend='nccl',  # Use NCCL for GPU
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            
            # Set device
            torch.cuda.set_device(local_rank)
            
            print(f"[Rank {rank}] Initialized DDP on GPU {local_rank}")
    
    def setup_multi_gpu_model(self, model):
        """Wrap model for multi-GPU training."""
        
        if self.multi_gpu_mode == 'ddp':
            # DistributedDataParallel (recommended)
            if self.local_rank != -1:
                model = model.to(self.device)
                model = DDP(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False  # Set to True if needed
                )
                if self.is_main_process:
                    print(f"✓ Using DistributedDataParallel on {self.world_size} GPUs")
            
        elif self.multi_gpu_mode == 'dp':
            # DataParallel (simpler but slower)
            if self.gpu_ids and len(self.gpu_ids) > 1:
                model = model.to(self.device)
                model = DP(model, device_ids=self.gpu_ids)
                print(f"✓ Using DataParallel on GPUs: {self.gpu_ids}")
            else:
                model = model.to(self.device)
                print(f"⚠ Only 1 GPU specified, using single GPU mode")
        
        else:
            # Single GPU
            model = model.to(self.device)
        
        return model
    
    def _print_multi_gpu_config(self):
        """Print multi-GPU configuration."""
        print("\n" + "="*80)
        print("MULTI-GPU CONFIGURATION")
        print("="*80)
        
        if self.multi_gpu_mode == 'ddp':
            print(f"Mode: DistributedDataParallel (DDP)")
            print(f"World Size: {self.world_size} GPUs")
            print(f"Local Rank: {self.local_rank}")
            print(f"Backend: NCCL")
        elif self.multi_gpu_mode == 'dp':
            print(f"Mode: DataParallel (DP)")
            print(f"GPU IDs: {self.gpu_ids}")
            print(f"Number of GPUs: {len(self.gpu_ids)}")
        else:
            print(f"Mode: Single GPU")
        
        print(f"Main Process: {self.is_main_process}")
        print("="*80 + "\n")
    
    def _print_config(self):
        """Override to only print on main process."""
        if self.is_main_process:
            super()._print_config()
    
    def train_epoch(self, epoch):
        """Train for one epoch with multi-GPU support."""
        
        # Set epoch for DistributedSampler
        if self.multi_gpu_mode == 'ddp' and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        return super().train_epoch(epoch)
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save checkpoint only on main process."""
        if self.is_main_process:
            # Get underlying model (unwrap DDP/DP)
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': {
                    'current_epoch': self.scheduler.current_epoch,
                    'base_lr': self.scheduler.base_lr
                },
                'val_acc': val_acc,
                'history': self.history,
                'config': {
                    'learning_rate': self.learning_rate,
                    'num_epochs': self.num_epochs,
                    'gradient_accumulation_steps': self.gradient_accumulation_steps,
                    'use_amp': self.use_amp,
                    'multi_gpu_mode': self.multi_gpu_mode,
                    'world_size': self.world_size
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
    
    def cleanup_ddp(self):
        """Cleanup distributed training."""
        if self.multi_gpu_mode == 'ddp' and dist.is_initialized():
            dist.destroy_process_group()


def create_distributed_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=32,
    num_workers=4,
    world_size=1,
    rank=0
):
    """Create dataloaders with DistributedSampler for DDP."""
    from torch.utils.data import DataLoader
    
    # Training sampler (shuffle via DistributedSampler)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    
    # Validation sampler (no shuffle)
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Test sampler (no shuffle)
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ============================================================================
# DDP TRAINING FUNCTION (for torch.distributed.launch)
# ============================================================================

def train_ddp(local_rank, args):
    """Training function for DistributedDataParallel.
    
    This is called by each process in distributed training.
    """
    from data_loader import MultiChannelHARDataset
    from models.encoder_cls import EncoderClassifier
    import torchvision.transforms as transforms
    
    # Setup
    world_size = args.world_size
    
    # Create datasets
    data_dir = args.data_dir
    mode = args.mode
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = MultiChannelHARDataset(
        f'{data_dir}/train_multichannel_images.pt',
        mode=mode,
        transform=train_transform
    )
    
    val_dataset = MultiChannelHARDataset(
        f'{data_dir}/val_multichannel_images.pt',
        mode=mode,
        transform=test_transform
    )
    
    test_dataset = MultiChannelHARDataset(
        f'{data_dir}/test_multichannel_images.pt',
        mode=mode,
        transform=test_transform
    )
    
    # Create distributed dataloaders
    train_loader, val_loader, test_loader = create_distributed_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        world_size=world_size,
        rank=local_rank
    )
    
    # Create model
    model = EncoderClassifier(
        num_classes=6,
        pretrained_encoder_path=args.encoder_path,
        freeze_encoder=args.freeze_encoder,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        image_size=224
    )
    
    # Create trainer
    trainer = MultiGPUTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=f'cuda:{local_rank}',
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        early_stopping_patience=args.early_stopping_patience,
        save_dir=args.save_dir,
        use_amp=not args.no_amp,
        log_interval=args.log_interval,
        multi_gpu_mode='ddp',
        local_rank=local_rank,
        world_size=world_size
    )
    
    # Train
    results = trainer.train()
    
    # Cleanup
    trainer.cleanup_ddp()
    
    return results


# ============================================================================
# DATAPARALLEL TRAINING (simpler, no launch script needed)
# ============================================================================

def train_dataparallel(args):
    """Training with DataParallel (simpler but slower than DDP)."""
    from data_loader import create_har_dataloaders
    from models.encoder_cls import EncoderClassifier
    
    # Create dataloaders (regular, not distributed)
    train_loader, val_loader, test_loader = create_har_dataloaders(
        data_dir=args.data_dir,
        mode=args.mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=True
    )
    
    # Create model
    model = EncoderClassifier(
        num_classes=6,
        pretrained_encoder_path=args.encoder_path,
        freeze_encoder=args.freeze_encoder,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        image_size=224
    )
    
    # Get GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
    
    # Create trainer
    trainer = MultiGPUTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=f'cuda:{gpu_ids[0]}',
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        early_stopping_patience=args.early_stopping_patience,
        save_dir=args.save_dir,
        use_amp=not args.no_amp,
        log_interval=args.log_interval,
        multi_gpu_mode='dp',
        gpu_ids=gpu_ids
    )
    
    # Train
    results = trainer.train()
    
    return results


if __name__ == '__main__':
    print("This module provides multi-GPU training support.")
    print("\nUsage:")
    print("  1. DistributedDataParallel (recommended):")
    print("     python main_multi_gpu.py --multi-gpu ddp")
    print("     or")
    print("     torchrun --nproc_per_node=4 main_multi_gpu.py --multi-gpu ddp")
    print("\n  2. DataParallel (simpler):")
    print("     python main_multi_gpu.py --multi-gpu dp")
